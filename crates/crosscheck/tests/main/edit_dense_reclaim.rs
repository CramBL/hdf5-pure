#![cfg(feature = "__hdf5-1.10")]
//! Interoperability tests for dense-storage reclamation.
//!
//! Dense attribute and dense group version 2 B-tree indexes become reusable when their owner
//! dies. Their fractal heaps stay allocated. These cases also cover surviving group hard links
//! and persistent FSM/PAGE files accepted by libhdf5.

use std::path::Path;

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5::plist::group_create::{GroupCreate, GroupCreateBuilder};
use hdf5::{IndexType, IterationOrder};
use hdf5_pure::{Error, File, FormatError};
use tempfile::tempdir;
use test_util_hdf5::absence;
use test_util_hdf5::creation_order::{self, Indexing};

fn write_tracked_with_persistence(path: &Path, names: &[String], indexed: Indexing, persist: bool) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| {
            let lower = if persist {
                LibraryVersion::V110
            } else {
                LibraryVersion::V18
            };
            p.libver_bounds(lower, LibraryVersion::latest())
        })
        .with_fcpl(|p| {
            p.attr_creation_order(indexed.attr_order());
            if persist {
                p.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: true,
                    threshold: 1,
                })
            } else {
                p
            }
        })
        .create(path)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    let group = file
        .create_group_builder()
        .set_gcpl(&creation_order::group_properties(indexed, false))
        .create("g")
        .expect("create group");
    let dataset = file
        .new_dataset::<i32>()
        .with_dcpl(|p| p.attr_creation_order(indexed.attr_order()))
        .shape([4])
        .create("d")
        .expect("create dataset");
    let owners: [&hdf5::Location; 2] = [&group, &dataset];
    for owner in owners {
        for (i, name) in names.iter().enumerate() {
            owner
                .new_attr::<i32>()
                .shape(())
                .create(name.as_str())
                .unwrap_or_else(|e| panic!("create attribute {name}: {e}"))
                .write_scalar(&(i as i32))
                .unwrap_or_else(|e| panic!("write attribute {name}: {e}"));
        }
    }
    file.close().unwrap();
}

fn object_location(path: &Path, object: &str) -> (hdf5::File, hdf5::Location) {
    let file = hdf5::File::open(path)
        .unwrap_or_else(|e| panic!("the C library opens {}: {e}", path.display()));
    let location = if object == "g" {
        (*file
            .group(object)
            .unwrap_or_else(|e| panic!("the C library opens /{object}: {e}")))
        .clone()
    } else {
        (**file
            .dataset(object)
            .unwrap_or_else(|e| panic!("the C library opens /{object}: {e}")))
        .clone()
    };
    (file, location)
}

fn attribute_name_by_creation_index(path: &Path, object: &str, n: u64) -> String {
    let (_file, location) = object_location(path, object);
    location
        .attr_by_index(IndexType::CreationOrder, IterationOrder::Increasing, n)
        .unwrap_or_else(|e| panic!("the C library opens attribute {n} by creation order: {e}"))
        .name()
}

fn attribute_names_by_name(path: &Path, object: &str) -> Vec<String> {
    let (_file, location) = object_location(path, object);
    location
        .attr_names_by(IndexType::Name, IterationOrder::Increasing)
        .unwrap_or_else(|e| panic!("the C library iterates attributes: {e}"))
}

#[test]
fn c_library_accepts_persisted_reclaimed_dense_attribute_indexes() {
    let dir = tempdir().unwrap();
    let attribute_names = creation_order::names(12);

    for indexed in [Indexing::No, Indexing::Yes] {
        let suffix = if indexed == Indexing::Yes {
            "indexed"
        } else {
            "name_only"
        };
        let path = dir
            .path()
            .join(format!("dense_attribute_reclaim_{suffix}.h5"));
        write_tracked_with_persistence(&path, &attribute_names, indexed, true);

        if indexed == Indexing::Yes {
            assert_eq!(
                attribute_name_by_creation_index(&path, "d", 0),
                attribute_names[0]
            );
        }
        let free_before: u64 = File::open(&path)
            .unwrap()
            .persisted_free_space()
            .unwrap()
            .iter()
            .map(|&(_, len)| len)
            .sum();

        {
            let session = File::open_rw(&path).unwrap();
            session.root().delete("d").unwrap();
            session.commit().unwrap();
        }

        let pure = File::open(&path).unwrap();
        let err = pure.dataset("d").unwrap_err();
        let Error::Format(FormatError::PathNotFound(missing)) = &err else {
            panic!("expected PathNotFound for d, got {err:?}");
        };
        assert_eq!(missing, "d");
        assert_eq!(pure.group("g").unwrap().attrs().unwrap().len(), 12);
        let free_after: u64 = pure
            .persisted_free_space()
            .unwrap()
            .iter()
            .map(|&(_, len)| len)
            .sum();
        assert!(
            free_after > free_before,
            "deleting the dense dataset must add its proven storage to persisted free space"
        );
        drop(pure);

        let c = hdf5::File::open(&path).unwrap();
        absence::assert_libhdf5_absent(&c.dataset("d").unwrap_err(), "d");
        assert_eq!(attribute_names_by_name(&path, "g").len(), 12);
        assert!(
            c.free_space() >= free_after,
            "libhdf5 must accept the free-space managers written after dense-index reclamation"
        );
        drop(c);

        {
            let c = hdf5::File::open_rw(&path).unwrap();
            c.new_dataset::<i32>()
                .shape([4])
                .create("after")
                .unwrap()
                .write(&[4, 3, 2, 1])
                .unwrap();
            c.close().unwrap();
        }
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("after")
                .unwrap()
                .read_i32()
                .unwrap(),
            vec![4, 3, 2, 1]
        );
    }
}

/// Writes a persistent file whose `/g` and `/g/nested` groups both use dense link storage.
///
/// `Indexing::No` uses ordinary name indexing, while `Indexing::Yes` also tracks and indexes link
/// creation order. `/g/shared` has a root hard link so deleting the group exercises child lifetime
/// independently of link-index ownership.
fn dense_link_gcpl(indexed: Indexing) -> GroupCreate {
    let mut builder = GroupCreateBuilder::new();
    if indexed == Indexing::Yes {
        builder.link_creation_order(Indexing::Yes.link_order());
    }
    builder
        .finish()
        .expect("a dense-link group creation property list")
}

fn write_dense_link_reclaim_fixture(
    path: &Path,
    indexed: Indexing,
    paged: bool,
    alias_group: bool,
) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .with_fcpl(|p| {
            p.file_space_strategy(CStrategy::FreeSpaceManager {
                paged,
                persist: true,
                threshold: 1,
            })
        })
        .create(path)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    let group = file
        .create_group_builder()
        .set_gcpl(&dense_link_gcpl(indexed))
        .create("g")
        .expect("create dense parent group");
    for i in 0i32..12 {
        let link_name = format!("d{i}");
        group
            .new_dataset::<i32>()
            .shape([1])
            .create(link_name.as_str())
            .unwrap_or_else(|e| panic!("create dense-group child {link_name}: {e}"))
            .write(&[i])
            .unwrap_or_else(|e| panic!("write dense-group child {link_name}: {e}"));
    }
    let nested = file
        .create_group_builder()
        .set_gcpl(&dense_link_gcpl(indexed))
        .create("g/nested")
        .expect("create nested dense group");
    for i in 0i32..12 {
        let link_name = format!("n{i}");
        nested
            .new_dataset::<i32>()
            .shape([1])
            .create(link_name.as_str())
            .unwrap_or_else(|e| panic!("create nested child {link_name}: {e}"))
            .write(&[100 + i])
            .unwrap_or_else(|e| panic!("write nested child {link_name}: {e}"));
    }
    group
        .new_dataset::<i32>()
        .shape([4])
        .create("shared")
        .unwrap()
        .write(&[4, 3, 2, 1])
        .unwrap();
    file.link_hard("/g/shared", "/shared_survivor").unwrap();
    if alias_group {
        file.link_hard("/g", "/alias").unwrap();
    }
    file.new_dataset::<i32>()
        .shape([4])
        .create("keep")
        .unwrap()
        .write(&[1, 2, 3, 4])
        .unwrap();
    file.close().unwrap();
}

fn persistent_free(path: &Path) -> Vec<(u64, u64)> {
    File::open(path).unwrap().persisted_free_space().unwrap()
}

fn free_bytes(sections: &[(u64, u64)]) -> u64 {
    sections.iter().map(|&(_, len)| len).sum()
}

fn overlaps_free(sections: &[(u64, u64)], addr: u64, len: u64) -> bool {
    let Some(end) = addr.checked_add(len) else {
        return true;
    };
    sections.iter().any(|&(free_addr, free_len)| {
        let Some(free_end) = free_addr.checked_add(free_len) else {
            return true;
        };
        free_addr < end && addr < free_end
    })
}

/// Returns header and root-node spans for each dense-link type 5 or type 6 B-tree in `path`.
fn dense_link_tree_anchor_spans(path: &Path) -> Vec<(u8, u64, u64)> {
    let bytes = std::fs::read(path).unwrap();
    let superblock = hdf5_pure_format::__private::parse_superblock(&bytes, 0).unwrap();
    let header_size = 22 + u64::from(superblock.offset_size) + u64::from(superblock.length_size);
    let mut spans = Vec::new();
    for at in 0..bytes.len().saturating_sub(5) {
        if bytes.get(at..at + 4) != Some(b"BTHD") || !matches!(bytes[at + 5], 5 | 6) {
            continue;
        }
        let header = hdf5_pure_format::__private::BTreeV2Header::parse(
            &bytes,
            at,
            superblock.offset_size,
            superblock.length_size,
        )
        .unwrap();
        spans.push((header.tree_type, u64::try_from(at).unwrap(), header_size));
        spans.push((
            header.tree_type,
            header.root_node_address.get(),
            u64::from(header.node_size),
        ));
    }
    assert!(
        !spans.is_empty(),
        "the fixture must contain dense-link B-trees"
    );
    spans
}

fn fractal_heap_headers(path: &Path) -> Vec<u64> {
    std::fs::read(path)
        .unwrap()
        .windows(4)
        .enumerate()
        .filter(|&(_, bytes)| bytes == b"FRHP")
        .map(|(at, _)| u64::try_from(at).unwrap())
        .collect()
}

#[test]
fn c_library_accepts_reclaimed_dense_group_indexes() {
    let dir = tempdir().unwrap();
    for (indexed, paged) in [
        (Indexing::No, false),
        (Indexing::Yes, false),
        (Indexing::No, true),
        (Indexing::Yes, true),
    ] {
        let path = dir.path().join(format!(
            "dense_group_reclaim_{}_{}.h5",
            if indexed == Indexing::Yes {
                "indexed"
            } else {
                "name_only"
            },
            if paged { "page" } else { "fsm" }
        ));
        write_dense_link_reclaim_fixture(&path, indexed, paged, false);
        let tree_anchors = dense_link_tree_anchor_spans(&path);
        assert!(tree_anchors.iter().any(|&(tree_type, _, _)| tree_type == 5));
        assert_eq!(
            tree_anchors.iter().any(|&(tree_type, _, _)| tree_type == 6),
            indexed == Indexing::Yes
        );
        let heap_headers = fractal_heap_headers(&path);
        assert!(
            heap_headers.len() >= 2,
            "the parent and nested group must both use dense storage"
        );
        let free_before = persistent_free(&path);

        {
            let session = File::open_rw(&path).unwrap();
            session.root().delete("g").unwrap();
            session.commit().unwrap();
        }

        let free_after = persistent_free(&path);
        assert!(
            free_bytes(&free_after) > free_bytes(&free_before),
            "deleting the dense subtree must return proven storage"
        );
        // Rewriting the persisted free-space managers may consume a reclaimed index extent in
        // this same commit, so the original B-tree addresses need not remain listed as free.
        for heap in heap_headers {
            assert!(
                !overlaps_free(&free_after, heap, 4),
                "dense-link fractal-heap storage remains allocated"
            );
        }

        let pure = File::open(&path).unwrap();
        let err = pure.group("g").unwrap_err();
        let Error::Format(FormatError::PathNotFound(missing)) = &err else {
            panic!("expected PathNotFound for g, got {err:?}");
        };
        assert_eq!(missing, "g");
        assert_eq!(
            pure.dataset("shared_survivor").unwrap().read_i32().unwrap(),
            vec![4, 3, 2, 1]
        );
        drop(pure);

        let c = hdf5::File::open(&path).unwrap();
        absence::assert_libhdf5_absent(&c.group("g").unwrap_err(), "g");
        assert_eq!(
            c.dataset("shared_survivor")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            vec![4, 3, 2, 1]
        );
        assert_eq!(
            c.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
            vec![1, 2, 3, 4]
        );
        assert!(
            c.free_space() >= free_bytes(&free_after),
            "libhdf5 must accept the persisted managers after dense-group reclamation"
        );
    }
}

#[test]
fn a_surviving_dense_group_hard_link_keeps_its_indexes_live() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dense_group_surviving_alias.h5");
    write_dense_link_reclaim_fixture(&path, Indexing::Yes, false, true);
    let index_anchors = dense_link_tree_anchor_spans(&path);
    let heap_headers = fractal_heap_headers(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g").unwrap();
        session.commit().unwrap();
    }
    let free_with_alias = persistent_free(&path);
    for &(_, addr, len) in &index_anchors {
        assert!(
            !overlaps_free(&free_with_alias, addr, len),
            "a surviving hard link keeps the dense group's index storage live"
        );
    }
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("alias/shared")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![4, 3, 2, 1]
    );

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("alias").unwrap();
        session.commit().unwrap();
    }
    let free_without_alias = persistent_free(&path);
    assert!(
        free_bytes(&free_without_alias) > free_bytes(&free_with_alias),
        "the final hard link makes the dense group's owned storage reclaimable"
    );
    // The commit may place its rewritten free-space-manager metadata into one of the
    // index extents it just reclaimed, so the physical index addresses need not remain
    // listed as free after the commit. The increase above is the persisted-space invariant.
    for heap in heap_headers {
        assert!(
            !overlaps_free(&free_without_alias, heap, 4),
            "the dense-link fractal heap stays allocated after the group dies"
        );
    }
    assert_eq!(
        hdf5::File::open(&path)
            .unwrap()
            .dataset("shared_survivor")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        vec![4, 3, 2, 1]
    );
}
