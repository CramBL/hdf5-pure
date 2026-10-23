#![cfg(feature = "__hdf5-1.10")]
//! Interoperability tests for dense heap reads and live storage protection.
//!
//! The fixtures include managed blocks, huge-object indexes and payloads, and internal free-space
//! managers. Their allocations are decoded from serialized metadata and checked before and after
//! an unrelated edit.

use std::path::Path;

use hdf5_pure::{AttrValue, File};
use hdf5_pure_core::__private::DecodedStoredAddress;
use hdf5_pure_format::__private::FractalHeapHeader;

use test_util::fractal_heap;
use test_util_hdf5::heap_interop;
use test_util_hdf5::storage;

use crate::heap_allocations::{AllocationClass, Heap};

/// Returns a unique deterministic name of about `len` bytes that starts with `d{i}_`.
fn long_name(i: usize, len: usize) -> String {
    let prefix = format!("d{i}_");
    let pad = len.saturating_sub(prefix.len());
    format!("{prefix}{}", "x".repeat(pad))
}

/// Writes group `g` with one single-`i32` dataset per name and value `i`.
///
/// The latest format makes the C library store the links densely in a fractal heap.
fn write_group(path: &Path, names: &[String]) {
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_latest())
        .create(path)
        .unwrap();
    let g = file.create_group("g").unwrap();
    for (i, name) in names.iter().enumerate() {
        g.new_dataset::<i32>()
            .shape((1,))
            .create(name.as_str())
            .unwrap()
            .write(&[i as i32])
            .unwrap();
    }
    file.close().unwrap();
}

/// Checks that a huge-only heap's allocations stay unavailable and returns the index depth.
fn assert_huge_only_storage_unavailable(path: &Path) -> u16 {
    let bytes = std::fs::read(path).unwrap();
    let heap_headers = fractal_heap::header_offsets(&bytes);
    assert_eq!(
        heap_headers.len(),
        1,
        "fixture should contain one fractal heap"
    );
    let heap_at = heap_headers[0];
    let heap = FractalHeapHeader::parse(&bytes, heap_at, 8, 8).unwrap();
    assert_eq!(heap.managed_objects_count, 0);
    assert_eq!(heap.allocated_managed_space, 0);
    assert_eq!(heap.root_block_address.get(), u64::MAX);
    assert_eq!(
        heap.managed_block_free_space_manager_address.get(),
        u64::MAX
    );
    assert!(heap.huge_objects_count > 0);

    let heap_at = u64::try_from(heap_at).unwrap();
    let inspected = Heap::new(&bytes, heap_at, &heap);
    assert!(
        inspected
            .allocations
            .iter()
            .any(|allocation| allocation.class() == AllocationClass::RawData)
    );
    assert_live_heap_storage(path, &inspected);
    inspected.huge_index_depth.unwrap()
}

fn assert_live_heap_storage(path: &Path, heap: &Heap<'_>) {
    let file = File::open_rw(path).unwrap();
    let allocations = heap
        .allocations
        .iter()
        .map(|allocation| allocation.physical_extent(&file))
        .collect::<Vec<_>>();
    storage::assert_unavailable(&file, &allocations);
    file.root()
        .create_dataset("probe", |builder| {
            builder.with_i32_data(&[19; 32]);
        })
        .unwrap();
    file.commit().unwrap();
    storage::assert_unavailable(&file, &allocations);
    file.close().unwrap();
    heap_interop::assert_edit_round_trip(path);
}

#[test]
fn reads_dense_links_stored_as_huge_objects() {
    let dir = tempfile::tempdir().unwrap();
    // 4 KiB names push every link message just past the heap's 4096-byte managed
    // limit, so each is a huge object; 40 entries give the huge-objects B-tree
    // several records to search.
    let names: Vec<String> = (0..40).map(|i| long_name(i, 4096)).collect();
    let path = dir.path().join("huge_links.h5");
    write_group(&path, &names);
    heap_interop::assert_links_resolve(&path, &names);
}

#[test]
fn c_written_dense_link_huge_storage_stays_unavailable() {
    let dir = tempfile::tempdir().unwrap();
    let names: Vec<String> = (0..40).map(|i| long_name(i, 5000)).collect();
    let path = dir.path().join("owned_huge_links.h5");
    write_group(&path, &names);

    assert!(
        assert_huge_only_storage_unavailable(&path) > 0,
        "fixture should force a multi-node huge-object index"
    );
    heap_interop::assert_links_resolve(&path, &names);
}

#[test]
fn reads_dense_links_mixing_managed_and_huge() {
    let dir = tempfile::tempdir().unwrap();
    // Alternate short (managed) and very long (huge) names within one dense
    // group, so both heap-ID types are decoded from the same heap.
    let names: Vec<String> = (0..40)
        .map(|i| {
            if i % 2 == 0 {
                format!("short_{i}")
            } else {
                long_name(i, 5000)
            }
        })
        .collect();
    let path = dir.path().join("mixed_links.h5");
    write_group(&path, &names);
    heap_interop::assert_links_resolve(&path, &names);
}

#[test]
fn reads_dense_attributes_stored_as_huge_objects() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge_attrs.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_latest())
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape((1,))
            .create("data")
            .unwrap();
        ds.write(&[7i32]).unwrap();
        // Enough attributes to force dense storage, each with a ~5 KiB name so
        // the whole attribute message is stored as a huge object.
        for i in 0..30 {
            let name = format!("a{i}_{}", "y".repeat(5000));
            ds.new_attr::<i64>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .write_scalar(&(i as i64))
                .unwrap();
        }
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("data").unwrap();
    assert_eq!(ds.read_i32().unwrap(), vec![7]);

    let attrs = ds.attrs().unwrap();
    assert_eq!(attrs.len(), 30, "all dense attributes resolved");
    for (name, value) in &attrs {
        // Name is "a{i}_yyyy..."; the value written was `i`.
        let i: i64 = name[1..name.find('_').unwrap()].parse().unwrap();
        assert_eq!(*value, AttrValue::I64(i), "attribute {name}");
    }
}

#[test]
fn c_written_dense_attribute_huge_storage_stays_unavailable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("owned_huge_attrs.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_latest())
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape((1,))
            .create("data")
            .unwrap();
        ds.write(&[7i32]).unwrap();
        for i in 0..40 {
            let name = format!("a{i}_{}", "y".repeat(5000));
            ds.new_attr::<i64>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .write_scalar(&(i as i64))
                .unwrap();
        }
        file.close().unwrap();
    }

    assert!(
        assert_huge_only_storage_unavailable(&path) > 0,
        "fixture should force a multi-node huge-object index"
    );
    let reference = hdf5::File::open(&path).unwrap();
    let dataset = reference.dataset("data").unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [7]);
    for index in 0..40 {
        let attr_name = format!("a{index}_{}", "y".repeat(5000));
        assert_eq!(
            dataset
                .attr(&attr_name)
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            index
        );
    }
}

#[test]
fn c_written_internal_heap_manager_storage_stays_unavailable() {
    let dir = tempfile::tempdir().unwrap();
    let names: Vec<String> = (0..16).map(|i| format!("short_{i}")).collect();
    let path = dir.path().join("owned_managed_links.h5");
    write_group(&path, &names);

    let bytes = std::fs::read(&path).unwrap();
    let heap_headers = fractal_heap::header_offsets(&bytes);
    assert_eq!(heap_headers.len(), 1);
    let heap_at = heap_headers[0];
    let heap = FractalHeapHeader::parse(&bytes, heap_at, 8, 8).unwrap();
    assert!(heap.managed_objects_count > 0);
    assert_ne!(
        heap.managed_block_free_space_manager_address.get(),
        u64::MAX,
        "fixture must have an internal manager"
    );

    let inspected = Heap::new(&bytes, u64::try_from(heap_at).unwrap(), &heap);
    let manager = inspected.manager.as_ref().unwrap();
    let DecodedStoredAddress::Defined(list_address) = manager.section_list_addr() else {
        panic!("expected an allocated section list")
    };
    assert!(manager.section_list_allocated() > 0);
    assert!(
        inspected
            .extents()
            .any(|extent| extent == (list_address.get(), manager.section_list_allocated()))
    );
    assert_eq!(heap.huge_objects_count, 0);
    assert_live_heap_storage(&path, &inspected);
    heap_interop::assert_links_resolve(&path, &names);
}
