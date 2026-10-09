#![cfg(feature = "__hdf5-1.10")]
//! Crosschecks dense links and attributes whose messages use fractal-heap huge-object storage.
//!
//! A link or attribute message larger than the heap's maximum managed-object size is stored as a
//! huge object. The reference C library writes these fixtures, and hdf5-pure resolves them through
//! the heap's huge-object version 2 B-tree with both buffered and streaming readers. Ownership
//! crosschecks independently inspect that B-tree and require every referenced metadata allocation
//! and huge-object payload to lie inside the heap ownership result.

use hdf5_pure::{AttrValue, File};
use hdf5_pure_format::__private::{
    BTREE_V2_HUGE_OBJECT, BTREE_V2_HUGE_OBJECT_DIRECT, BTreeV2Header, BTreeV2NodeInfo,
    BTreeV2Record, FractalHeapHeader, LengthWidth, OffsetWidth, btree_v2_header_size,
    parse_btree_v2_internal_child_pointers, parse_btree_v2_leaf_records,
};
use std::num::NonZeroU16;
use std::path::Path;
use tempfile::tempdir;
use test_util::fractal_heap;

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

fn extent_contains(extents: &[(u64, u64)], address: u64, length: u64) -> bool {
    let Some(end) = address.checked_add(length) else {
        return false;
    };
    extents.iter().any(|&(start, len)| {
        start <= address
            && start
                .checked_add(len)
                .is_some_and(|extent_end| end <= extent_end)
    })
}

/// Verifies ownership of a heap whose messages are all huge objects written by the reference C
/// library.
///
/// The format readers independently parse the heap header and walk its huge-object index. Every
/// B-tree metadata allocation and huge-object payload they reference must lie inside the ownership
/// result. The returned index depth lets callers verify that a fixture contains internal nodes.
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
    let extents = hdf5_pure::__fractal_heap_storage_extents(&bytes, heap_at, 8, 8).unwrap();
    let heap_len = u64::try_from(FractalHeapHeader::serialized_size(
        OffsetWidth::Eight,
        LengthWidth::Eight,
    ))
    .unwrap();
    assert!(extent_contains(&extents, heap_at, heap_len));

    let tree_at = heap.btree_huge_objects_address.get();
    let tree = BTreeV2Header::parse(&bytes, usize::try_from(tree_at).unwrap(), 8, 8).unwrap();
    assert!(
        tree.tree_type == BTREE_V2_HUGE_OBJECT || tree.tree_type == BTREE_V2_HUGE_OBJECT_DIRECT
    );
    assert!(extent_contains(
        &extents,
        tree_at,
        u64::try_from(btree_v2_header_size(OffsetWidth::Eight, LengthWidth::Eight)).unwrap()
    ));

    let node_info = BTreeV2NodeInfo::compute(tree.node_size, tree.record_size, 8, tree.depth);
    let mut records = Vec::new();
    inspect_huge_index_node(
        &bytes,
        &extents,
        &tree,
        &node_info,
        tree.root_node_address.get(),
        tree.num_records_in_root,
        tree.depth,
        &mut records,
    );
    assert_eq!(
        u64::try_from(records.len()).unwrap(),
        heap.huge_objects_count
    );
    assert_eq!(tree.total_records, heap.huge_objects_count);
    for record in records {
        let (address, length) = if tree.tree_type == BTREE_V2_HUGE_OBJECT {
            let record = record
                .huge_object(OffsetWidth::Eight, LengthWidth::Eight)
                .unwrap();
            (record.address.get(), record.length)
        } else {
            let record = record
                .huge_object_direct(OffsetWidth::Eight, LengthWidth::Eight)
                .unwrap();
            (record.address.get(), record.length)
        };
        assert!(
            extent_contains(&extents, address, length),
            "huge object {address:#x}+{length} is outside the ownership result"
        );
    }

    tree.depth
}

#[allow(clippy::too_many_arguments)]
fn inspect_huge_index_node(
    bytes: &[u8],
    extents: &[(u64, u64)],
    tree: &BTreeV2Header,
    node_info: &BTreeV2NodeInfo,
    address: u64,
    records_in_node: u16,
    depth: u16,
    records: &mut Vec<BTreeV2Record>,
) {
    assert!(extent_contains(extents, address, u64::from(tree.node_size)));
    let at = usize::try_from(address).unwrap();
    let end = at + usize::try_from(tree.node_size).unwrap();
    let node = &bytes[at..end];
    let signature: &[u8; 4] = if depth == 0 { b"BTLF" } else { b"BTIN" };
    assert_eq!(&node[..4], signature);
    assert_eq!(node[4], 0);
    assert_eq!(node[5], tree.tree_type);

    let Some(depth) = NonZeroU16::new(depth) else {
        records.extend(
            parse_btree_v2_leaf_records(node, 0, records_in_node, tree.record_size).unwrap(),
        );
        return;
    };

    let children = parse_btree_v2_internal_child_pointers(
        node,
        records_in_node,
        depth,
        tree.record_size,
        8,
        node_info,
    )
    .unwrap();
    let record_size = usize::from(tree.record_size);
    for (i, (child_address, child_records)) in children.into_iter().enumerate() {
        inspect_huge_index_node(
            bytes,
            extents,
            tree,
            node_info,
            child_address.get(),
            child_records,
            depth.get() - 1,
            records,
        );
        if i < usize::from(records_in_node) {
            let start = 6 + i * record_size;
            records.push(BTreeV2Record {
                data: node[start..start + record_size].to_vec(),
            });
        }
    }
}

#[test]
fn reads_dense_links_stored_as_huge_objects() {
    let dir = tempdir().unwrap();
    // 4 KiB names push every link message just past the heap's 4096-byte managed
    // limit, so each is a huge object; 40 entries give the huge-objects B-tree
    // several records to search.
    let names: Vec<String> = (0..40).map(|i| long_name(i, 4096)).collect();
    let path = dir.path().join("huge_links.h5");
    write_group(&path, &names);
    assert_links_resolve(&path, &names);
}

#[test]
fn owned_storage_covers_c_written_dense_link_huge_objects() {
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
fn owned_storage_covers_c_written_dense_attribute_huge_objects() {
    let dir = tempdir().unwrap();
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
fn owned_storage_refuses_c_written_managed_heap_with_internal_fsm() {
    let dir = tempdir().unwrap();
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
        "fixture should exercise the explicitly unsupported internal manager"
    );

    assert_eq!(
        hdf5_pure::__fractal_heap_storage_extents(&bytes, u64::try_from(heap_at).unwrap(), 8, 8,)
            .unwrap_err(),
        "UnsupportedOwnership(\"managed-space free-space manager metadata\")"
    );
}
