#![cfg(feature = "__hdf5-1.10")]
//! Crosschecks fractal-heap storage ownership against independently decoded allocations.
//!
//! libhdf5 produces dense links and attributes stored as huge objects, including
//! multi-node huge-object indexes. The hdf5-pure writer produces supported managed heaps with
//! direct and nested indirect roots. Each ownership crosscheck reconstructs the complete expected
//! extent set from the serialized structures and compares it with the ownership result.

use std::path::Path;

use hdf5_pure::{AttrValue, File, FileBuilder};
use hdf5_pure_core::__private::DecodedStoredAddress;
use hdf5_pure_format::__private::FractalHeapHeader;

use test_util::fractal_heap;

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

/// Verifies that every `g/{name}` resolves to dataset value `i` through both readers.
fn assert_links_resolve(path: &Path, names: &[String]) {
    let buffered = File::open(path).unwrap();
    let streaming = File::open_streaming(path).unwrap();
    for (i, name) in names.iter().enumerate() {
        let p = format!("g/{name}");
        assert_eq!(
            buffered.dataset(&p).unwrap().read_i32().unwrap(),
            vec![i as i32],
            "buffered link {i}"
        );
        assert_eq!(
            streaming.dataset(&p).unwrap().read_i32().unwrap(),
            vec![i as i32],
            "streaming link {i}"
        );
    }
}

/// Compares the collector's allocations with decoded fixture metadata and returns the index depth.
fn assert_huge_only_heap_ownership(path: &Path) -> u16 {
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
    let mut actual = hdf5_pure::__fractal_heap_storage_extents(&bytes, heap_at, 8, 8).unwrap();
    let inspected = Heap::new(&bytes, heap_at, &heap);
    assert!(
        inspected
            .allocations
            .iter()
            .any(|allocation| allocation.class() == AllocationClass::RawData)
    );
    actual.sort_unstable();
    assert!(
        actual.iter().copied().eq(inspected.extents()),
        "{actual:?} != {:?}",
        inspected.allocations
    );
    inspected.huge_index_depth.unwrap()
}

fn assert_pure_managed_heap_ownership(bytes: &[u8], root_is_direct: bool) -> usize {
    let heap_headers = fractal_heap::header_offsets(bytes);
    assert_eq!(heap_headers.len(), 1);
    let heap_at = u64::try_from(heap_headers[0]).unwrap();
    let heap = FractalHeapHeader::parse(bytes, usize::try_from(heap_at).unwrap(), 8, 8).unwrap();
    let inspected = Heap::new(bytes, heap_at, &heap);
    assert!(inspected.header.managed_objects_count > 0);
    assert_eq!(inspected.header.huge_objects_count, 0);
    assert_eq!(
        inspected
            .header
            .managed_block_free_space_manager_address
            .get(),
        u64::MAX
    );
    assert_eq!(
        inspected.header.current_rows_in_root_indirect_block == 0,
        root_is_direct
    );
    assert!(
        inspected
            .allocations
            .iter()
            .all(|allocation| allocation.class() == AllocationClass::Metadata)
    );
    let actual = hdf5_pure::__fractal_heap_storage_extents(bytes, heap_at, 8, 8).unwrap();
    assert!(
        actual.iter().copied().eq(inspected.extents()),
        "{actual:?} != {:?}",
        inspected.allocations
    );
    inspected.indirect_blocks
}

#[test]
fn owned_storage_exactly_matches_pure_written_direct_root() {
    let mut builder = FileBuilder::new();
    for i in 0..9 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().unwrap();
    assert_eq!(assert_pure_managed_heap_ownership(&bytes, true), 0);
}

#[test]
fn owned_storage_exactly_matches_pure_written_nested_indirect_root() {
    let mut builder = FileBuilder::new();
    for i in 0..12 {
        let text = char::from(b'a' + u8::try_from(i).unwrap())
            .to_string()
            .repeat(64_000);
        builder.set_attr(&format!("a{i:04}"), AttrValue::AsciiString(text));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().unwrap();
    let indirect_count = assert_pure_managed_heap_ownership(&bytes, false);
    assert!(
        indirect_count >= 2,
        "fixture should contain a nested managed indirect block"
    );
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
    assert_links_resolve(&path, &names);
}

#[test]
fn owned_storage_exactly_matches_c_written_dense_link_huge_objects() {
    let dir = tempfile::tempdir().unwrap();
    let names: Vec<String> = (0..40).map(|i| long_name(i, 5000)).collect();
    let path = dir.path().join("owned_huge_links.h5");
    write_group(&path, &names);

    assert!(
        assert_huge_only_heap_ownership(&path) > 0,
        "fixture should force a multi-node huge-object index"
    );
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
    assert_links_resolve(&path, &names);
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
fn owned_storage_exactly_matches_c_written_dense_attribute_huge_objects() {
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
        assert_huge_only_heap_ownership(&path) > 0,
        "fixture should force a multi-node huge-object index"
    );
}

#[test]
fn owned_storage_includes_c_written_managed_heap_with_internal_fsm() {
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

    let allocations =
        hdf5_pure::__fractal_heap_storage_extents(&bytes, u64::try_from(heap_at).unwrap(), 8, 8)
            .unwrap();
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
    assert!(
        allocations.iter().copied().eq(inspected.extents()),
        "{allocations:?} != {:?}",
        inspected.allocations
    );
}
