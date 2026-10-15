#![cfg(feature = "__hdf5-1.10")]
//! Interoperability tests for the reclamation of dense attribute and dense link storage.
//!
//! libhdf5 writes each file, [`File::open_rw`] edits it, and libhdf5 reads the result. The cases
//! cover the version 2 B-tree indexes and the fractal heap of a dead dense set, a hard link that
//! keeps a dense set alive, and files that persist their free space with and without pages.

use std::path::Path;

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5::plist::group_create::{GroupCreate, GroupCreateBuilder};
use hdf5::{IndexType, IterationOrder};
use hdf5_pure::AttrValue;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use hdf5_pure::MessageType;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::AttributeInfoMessage;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use hdf5_pure_format::__private::FreeSpaceManagerHeaderFrame;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::LinkMessage;
use hdf5_pure_format::__private::LinkTarget;
use hdf5_pure_format::__private::ObjectHeader;
use rstest::rstest;
use tempfile::tempdir;
use test_util::btree_v2;
use test_util::fractal_heap;
use test_util::free_space;
use test_util::range;
use test_util_hdf5::absence;
use test_util_hdf5::creation_order::{self, Indexing};

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
        creation_order::write_attribute_fixture(&path, &attribute_names, indexed, true);

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
    let superblock = hdf5_pure_format::__private::parse_superblock_with_widths(&bytes, 0)
        .unwrap()
        .0;
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
        let bytes = std::fs::read(&path).unwrap();
        let heap_allocations = heap_headers
            .iter()
            .flat_map(|&heap| {
                hdf5_pure::__fractal_heap_storage_extents(&bytes, heap, 8, 8).unwrap()
            })
            .collect::<Vec<_>>();
        assert!(heap_allocations.iter().any(|&(address, _)| {
            bytes.get(usize::try_from(address).unwrap()..usize::try_from(address).unwrap() + 4)
                == Some(free_space::SIGNATURE)
        }));
        assert!(heap_allocations.iter().any(|&(address, _)| {
            bytes.get(usize::try_from(address).unwrap()..usize::try_from(address).unwrap() + 4)
                == Some(free_space::SECTIONS_SIGNATURE)
        }));
        let free_before = persistent_free(&path);

        {
            let session = File::open_rw(&path).unwrap();
            session.root().delete("g").unwrap();
            session.commit().unwrap();
        }

        let released = released_regions(&path);
        for &(_, addr, len) in &tree_anchors {
            assert_eq!(
                range::covered_len(&released, &(addr..addr + len)),
                len,
                "the dense-link index block ({addr}, {len}) must be reclaimed"
            );
        }
        let free_after = persistent_free(&path);
        assert!(
            free_bytes(&free_after) > free_bytes(&free_before),
            "deleting the dense subtree must return proven storage"
        );
        for &(address, len) in &heap_allocations {
            let reclaim = if paged && len >= 4096 && address % 4096 == 0 {
                len / 4096 * 4096
            } else {
                len
            };
            assert_eq!(
                range::covered_len(&released, &(address..address + reclaim)),
                reclaim
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
    let bytes = std::fs::read(&path).unwrap();
    let owned_heaps = heap_headers
        .iter()
        .flat_map(|&heap| hdf5_pure::__fractal_heap_storage_extents(&bytes, heap, 8, 8).unwrap())
        .collect::<Vec<_>>();

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
    let released = released_regions(&path);
    for &(_, addr, len) in &index_anchors {
        assert_eq!(
            range::covered_len(&released, &(addr..addr + len)),
            len,
            "the dense-link index block ({addr}, {len}) must be reclaimed"
        );
    }
    for &(address, len) in &owned_heaps {
        assert_eq!(
            range::covered_len(&released, &(address..address + len)),
            len
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

struct DenseNames {
    links: Vec<String>,
    attributes: Vec<String>,
}

impl DenseNames {
    fn new() -> Self {
        Self {
            links: (0..10)
                .map(|i| format!("link_{i:02}_{}", "x".repeat(20)))
                .collect(),
            attributes: (0..10)
                .map(|i| format!("attr_{i:02}_{}", "x".repeat(40)))
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FileSpace {
    Managers,
    ManagersAfterUserblock,
    Pages(u64),
}

impl FileSpace {
    fn page_size(self) -> Option<u64> {
        match self {
            Self::Managers | Self::ManagersAfterUserblock => None,
            Self::Pages(page_size) => Some(page_size),
        }
    }
}

fn huge_name(prefix: &str, i: usize) -> String {
    format!("{prefix}{i}_{}", "z".repeat(5000))
}

fn create_dense_file(path: &Path, file_space: FileSpace) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .with_fcpl(|p| match file_space {
            FileSpace::Managers | FileSpace::ManagersAfterUserblock => {
                if file_space == FileSpace::ManagersAfterUserblock {
                    p.userblock(USERBLOCK);
                }
                p.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: true,
                    threshold: 1,
                })
            }
            FileSpace::Pages(page_size) => p
                .file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: true,
                    persist: true,
                    threshold: 0,
                })
                .file_space_page_size(page_size),
        })
        .create(path)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()))
}

fn add_dense_objects(file: &hdf5::File, names: &DenseNames, huge_objects: usize) {
    let group = file.create_group("g").expect("create dense group");
    for (i, link_name) in (0i32..).zip(&names.links) {
        group
            .new_dataset::<i32>()
            .shape([1])
            .create(link_name.as_str())
            .unwrap_or_else(|e| panic!("create dense-group child {link_name}: {e}"))
            .write(&[i])
            .unwrap_or_else(|e| panic!("write dense-group child {link_name}: {e}"));
    }
    for i in 0..huge_objects {
        group
            .new_dataset::<i32>()
            .shape([1])
            .create(huge_name("huge_link_", i).as_str())
            .unwrap_or_else(|e| panic!("create huge link {i}: {e}"));
    }
    add_dense_attributes(&group, names, huge_objects);
    add_dense_attribute_dataset(file, names, huge_objects);
}

fn add_dense_attribute_dataset(file: &hdf5::File, names: &DenseNames, huge_objects: usize) {
    let dataset = file
        .new_dataset::<i32>()
        .shape([4])
        .create("d")
        .expect("create dataset with dense attributes");
    add_dense_attributes(&dataset, names, huge_objects);
}

fn add_dense_attributes(owner: &hdf5::Location, names: &DenseNames, huge_objects: usize) {
    for (i, attr_name) in (0i64..).zip(&names.attributes) {
        owner
            .new_attr::<i64>()
            .shape(())
            .create(attr_name.as_str())
            .unwrap_or_else(|e| panic!("create attribute {attr_name}: {e}"))
            .write_scalar(&i)
            .unwrap_or_else(|e| panic!("write attribute {attr_name}: {e}"));
    }
    for i in 0..huge_objects {
        owner
            .new_attr::<i64>()
            .shape(())
            .create(huge_name("huge_attr_", i).as_str())
            .unwrap_or_else(|e| panic!("create huge attribute {i}: {e}"))
            .write_scalar(&0)
            .unwrap_or_else(|e| panic!("write huge attribute {i}: {e}"));
    }
}

// libhdf5 deletes the managed free-space manager of a heap that closes with no free section
// (`H5HF__space_close` in `H5HFspace.c`, 2.2.0), so a heap whose managed objects fill its root
// direct block exactly has none.
fn exactly_filling_names(dir: &Path, huge_objects: usize) -> DenseNames {
    let probe = dir.join("dense_probe.h5");
    let file = create_dense_file(&probe, FileSpace::Managers);
    add_dense_objects(&file, &DenseNames::new(), huge_objects);
    file.close().unwrap();
    let heaps = dense_heaps(&probe);
    let free_space = |start_block_size: u64| {
        let mut free: Vec<u64> = heaps
            .iter()
            .filter(|heap| heap.header.starting_block_size == start_block_size)
            .map(|heap| heap.header.free_space_in_managed_blocks)
            .collect();
        free.dedup();
        assert_eq!(free.len(), 1, "{start_block_size}-byte heaps: {free:?}");
        usize::try_from(free[0]).unwrap()
    };
    let mut names = DenseNames::new();
    let link_free = free_space(LINK_HEAP_START_BLOCK_SIZE);
    let attribute_free = free_space(ATTRIBUTE_HEAP_START_BLOCK_SIZE);
    names
        .links
        .last_mut()
        .unwrap()
        .push_str(&"y".repeat(link_free));
    names
        .attributes
        .last_mut()
        .unwrap()
        .push_str(&"y".repeat(attribute_free));
    names
}

fn write_dense_fixture(
    path: &Path,
    file_space: FileSpace,
    names: &DenseNames,
    huge_objects: usize,
    alias: bool,
) {
    let file = create_dense_file(path, file_space);
    add_dense_objects(&file, names, huge_objects);
    if alias {
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

fn framed_bytes(path: &Path) -> Vec<u8> {
    let base = File::open(path).unwrap().superblock().base_address.get();
    let mut bytes = std::fs::read(path).unwrap();
    bytes.drain(..usize::try_from(base).unwrap());
    bytes
}

struct DenseHeap {
    owner: String,
    address: u64,
    header: FractalHeapHeader,
}

fn dense_heaps(path: &Path) -> Vec<DenseHeap> {
    let bytes = framed_bytes(path);
    let object_header = |addr: u64| {
        ObjectHeader::parse(
            &bytes,
            AccessMode::ReadOnly,
            usize::try_from(addr).unwrap(),
            8,
            8,
        )
        .unwrap()
    };
    let file = File::open(path).unwrap();
    let root = object_header(file.superblock().root_group_address.get());
    let mut heaps: Vec<DenseHeap> = root
        .messages
        .iter()
        .filter(|message| message.msg_type == MessageType::LINK)
        .map(|message| LinkMessage::parse(&message.data, 8).unwrap())
        .filter_map(|link| match link.link_target {
            LinkTarget::Hard {
                object_header_address,
            } => Some((link.name, object_header_address.get())),
            LinkTarget::Soft { .. } | LinkTarget::External { .. } => None,
        })
        .flat_map(|(owner, address)| {
            object_header(address)
                .messages
                .into_iter()
                .map(move |message| (owner.clone(), message))
        })
        .filter_map(|(owner, message)| match message.msg_type {
            MessageType::LINK_INFO => LinkInfoMessage::parse(&message.data, 8)
                .unwrap()
                .fractal_heap_address
                .map(|address| (owner, address)),
            MessageType::ATTRIBUTE_INFO => AttributeInfoMessage::parse(&message.data, 8)
                .unwrap()
                .fractal_heap_address
                .map(|address| (owner, address)),
            _ => None,
        })
        .map(|(owner, address)| DenseHeap {
            owner,
            address: address.get(),
            header: FractalHeapHeader::parse(&bytes, usize::try_from(address.get()).unwrap(), 8, 8)
                .unwrap(),
        })
        .collect();
    heaps.sort_unstable_by_key(|heap| heap.address);
    heaps.dedup_by_key(|heap| heap.address);
    heaps
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PageClass {
    Metadata,
    RawData,
}

fn heap_allocations(
    path: &Path,
    huge_objects: usize,
    heaps: impl IntoIterator<Item = DenseHeap>,
) -> Vec<(u64, u64, PageClass)> {
    let bytes = framed_bytes(path);
    heaps
        .into_iter()
        .flat_map(|heap| {
            assert_eq!(
                heap.header.huge_objects_count,
                u64::try_from(huge_objects).unwrap()
            );
            hdf5_pure::__fractal_heap_storage_extents(&bytes, heap.address, 8, 8).unwrap()
        })
        .map(|(addr, len)| {
            let at = usize::try_from(addr).unwrap();
            let class = if HEAP_METADATA_SIGNATURES
                .iter()
                .any(|signature| &bytes[at..at + 4] == *signature)
            {
                PageClass::Metadata
            } else {
                PageClass::RawData
            };
            (addr, len, class)
        })
        .collect()
}

fn owned_heap_allocations(
    path: &Path,
    huge_objects: usize,
    managed_fsm: bool,
) -> Vec<(u64, u64, PageClass)> {
    let heaps = dense_heaps(path);
    assert_eq!(heaps.len(), 3, "the fixture holds three dense heaps");
    for heap in &heaps {
        assert_eq!(
            heap.header
                .managed_block_free_space_manager_address
                .is_undefined(OffsetWidth::Eight),
            !managed_fsm
        );
        if !managed_fsm {
            assert_eq!(heap.header.free_space_in_managed_blocks, 0);
        }
    }
    heap_allocations(path, huge_objects, heaps)
}

fn free_space_manager_blocks(path: &Path) -> Vec<(u64, u64)> {
    let bytes = framed_bytes(path);
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    File::open(path)
        .unwrap()
        .file_space_info()
        .unwrap()
        .manager_addrs
        .iter()
        .filter_map(|&addr| match addr {
            hdf5_pure::DecodedStoredAddress::Defined(stored) => Some(stored),
            hdf5_pure::DecodedStoredAddress::Undefined => None,
        })
        .flat_map(|stored| {
            let header = FreeSpaceManagerHeader::parse(
                widths,
                &bytes[usize::try_from(stored.get()).unwrap()..],
            )
            .unwrap();
            let mut blocks = vec![(
                stored.get(),
                hdf5_pure_format::__private::free_space_manager_header_len(widths),
            )];
            if !header.section_list_addr().is_undefined(OffsetWidth::Eight) {
                blocks.push((
                    header.section_list_addr().get(),
                    header.section_list_allocated(),
                ));
            }
            blocks
        })
        .collect()
}

fn released_regions(path: &Path) -> Vec<(u64, u64)> {
    let file_len = u64::try_from(framed_bytes(path).len()).unwrap();
    let mut released = persistent_free(path);
    released.extend(free_space_manager_blocks(path));
    released.push((file_len, u64::MAX - file_len));
    released
}

fn assert_released(
    page_size: Option<u64>,
    released: &[(u64, u64)],
    allocations: &[(u64, u64, PageClass)],
) {
    for &(addr, len, _) in allocations {
        let reclaimed = match page_size {
            Some(page_size) if is_page_aligned_large(page_size, addr, len) => {
                addr..addr + len / page_size * page_size
            }
            _ => addr..addr + len,
        };
        assert_eq!(
            range::covered_len(released, &reclaimed),
            reclaimed.end - reclaimed.start,
            "the heap allocation ({addr}, {len}) must be reclaimed over {reclaimed:?}"
        );
    }
}

// An allocation libhdf5 makes through the generic large manager is a page or more and begins on a
// page boundary (`H5MF__alloc_to_fs_type` and `H5MF__alloc_pagefs`, `H5MF.c`, libhdf5 2.2.0).
fn is_page_aligned_large(page_size: u64, addr: u64, len: u64) -> bool {
    len >= page_size && addr % page_size == 0
}

fn manager_sections(path: &Path, manager: usize) -> Vec<(u64, u64)> {
    let bytes = framed_bytes(path);
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let addr = File::open(path)
        .unwrap()
        .file_space_info()
        .unwrap()
        .manager_addrs[manager];
    let hdf5_pure::DecodedStoredAddress::Defined(stored) = addr else {
        return Vec::new();
    };
    let header =
        FreeSpaceManagerHeader::parse(widths, &bytes[usize::try_from(stored.get()).unwrap()..])
            .unwrap();
    if header.section_list_addr().is_undefined(OffsetWidth::Eight) {
        return Vec::new();
    }
    let at = usize::try_from(header.section_list_addr().get()).unwrap();
    hdf5_pure_format::__private::parse_section_info(widths, &bytes[at..], stored, &header)
        .unwrap()
        .into_iter()
        .map(|section| (section.addr.get(), section.size))
        .collect()
}

fn assert_page_classes(path: &Path, page_size: u64, allocations: &[(u64, u64, PageClass)]) {
    let metadata = manager_sections(path, SMALL_METADATA_MANAGER);
    let raw_data = manager_sections(path, SMALL_RAW_MANAGER);
    for &(addr, len, class) in allocations {
        let excluded: &[(&str, &[(u64, u64)])] = if is_page_aligned_large(page_size, addr, len) {
            &[("metadata", &metadata), ("raw-data", &raw_data)]
        } else {
            match class {
                PageClass::Metadata => &[("raw-data", &raw_data)],
                PageClass::RawData => &[("metadata", &metadata)],
            }
        };
        for &(manager, sections) in excluded {
            assert_eq!(
                range::covered_len(sections, &(addr..addr + len)),
                0,
                "the {class:?} heap allocation ({addr}, {len}) must not return to the small \
                 {manager} manager"
            );
        }
    }
}

fn assert_c_reads_after_reclaim(path: &Path) {
    let c = hdf5::File::open(path).unwrap();
    absence::assert_libhdf5_absent(&c.group("g").unwrap_err(), "g");
    absence::assert_libhdf5_absent(&c.dataset("d").unwrap_err(), "d");
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(c.free_space(), free_bytes(&persistent_free(path)));
}

#[rstest]
fn c_written_dense_heaps_are_reclaimed_with_their_owners(
    #[values(0, 2)] huge_objects: usize,
    #[values(
        FileSpace::Managers,
        FileSpace::ManagersAfterUserblock,
        FileSpace::Pages(DENSE_PAGE_SIZE),
        FileSpace::Pages(SMALL_PAGE_SIZE)
    )]
    file_space: FileSpace,
    #[values(false, true)] managed_fsm: bool,
) {
    let dir = tempdir().unwrap();
    let names = if managed_fsm {
        DenseNames::new()
    } else {
        exactly_filling_names(dir.path(), huge_objects)
    };
    let path = dir.path().join("dense_heap_reclaim.h5");
    write_dense_fixture(&path, file_space, &names, huge_objects, false);
    let allocations = owned_heap_allocations(&path, huge_objects, managed_fsm);
    if file_space == FileSpace::Pages(SMALL_PAGE_SIZE) {
        assert!(
            allocations
                .iter()
                .any(|&(addr, len, _)| is_page_aligned_large(SMALL_PAGE_SIZE, addr, len)),
            "the fixture holds a heap allocation of a page or more: {allocations:?}"
        );
    }
    assert_eq!(
        allocations
            .iter()
            .filter(|&&(_, _, class)| class == PageClass::RawData)
            .count(),
        3 * huge_objects
    );

    let session = File::open_rw(&path).unwrap();
    let ownership = session.__space_ownership().unwrap();
    assert!(ownership.analysis_complete);
    let base = session.superblock().base_address.get();
    for &(address, len, _) in &allocations {
        assert_eq!(
            range::covered_len(
                &ownership.live_metadata_space,
                &(base + address..base + address + len)
            ) + range::covered_len(
                &ownership.live_raw_space,
                &(base + address..base + address + len)
            ),
            len
        );
    }
    session.root().delete("g").unwrap();
    session.root().delete("d").unwrap();
    session.commit().unwrap();
    let ownership = session.__space_ownership().unwrap();
    assert!(ownership.analysis_complete);
    drop(session);
    let released = released_regions(&path);

    assert_released(file_space.page_size(), &released, &allocations);
    if let Some(page_size) = file_space.page_size() {
        assert_page_classes(&path, page_size, &allocations);
    }
    assert_c_reads_after_reclaim(&path);
}

#[rstest]
fn c_written_heap_ghost_sections_have_only_physical_manager_ownership(
    #[values(
        FileSpace::Managers,
        FileSpace::ManagersAfterUserblock,
        FileSpace::Pages(DENSE_PAGE_SIZE)
    )]
    file_space: FileSpace,
) {
    let dir = tempdir().unwrap();
    let names = DenseNames {
        links: (0..300)
            .map(|index| format!("link_{index:03}_{}", "x".repeat(100)))
            .collect(),
        attributes: (0..300)
            .map(|index| format!("attr_{index:03}_{}", "x".repeat(100)))
            .collect(),
    };
    let path = dir.path().join("dense_heap_ghost_sections.h5");
    write_dense_fixture(&path, file_space, &names, 0, false);
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        let group = file.group("g").unwrap();
        for link in &names.links[..150] {
            group.unlink(link).unwrap();
        }
        drop(group);
        file.close().unwrap();
    }
    let bytes = framed_bytes(&path);
    let heaps = dense_heaps(&path);
    assert_eq!(heaps.len(), 3);
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let managers = heaps
        .iter()
        .map(|heap| {
            FreeSpaceManagerHeaderFrame::parse(
                widths,
                &bytes[usize::try_from(
                    heap.header.managed_block_free_space_manager_address.get(),
                )
                .unwrap()..],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(
        managers
            .iter()
            .any(|manager| manager.unserialized_sections() > 0)
    );
    let allocations = heap_allocations(&path, 0, heaps);
    let session = File::open_rw(&path).unwrap();
    assert!(session.__space_ownership().unwrap().analysis_complete);
    session.root().delete("g").unwrap();
    session.root().delete("d").unwrap();
    session.commit().unwrap();
    assert!(session.__space_ownership().unwrap().analysis_complete);
    drop(session);
    let released = released_regions(&path);
    assert_released(file_space.page_size(), &released, &allocations);
    if let Some(page_size) = file_space.page_size() {
        assert_page_classes(&path, page_size, &allocations);
    }
    assert_c_reads_after_reclaim(&path);
}

#[rstest]
fn rewriting_a_c_written_dense_attribute_set_reclaims_its_old_heap(
    #[values("a", "d")] owner: &str,
    #[values(FileSpace::Managers, FileSpace::Pages(DENSE_PAGE_SIZE))] file_space: FileSpace,
) {
    let dir = tempdir().unwrap();
    let names = exactly_filling_names(dir.path(), 1);
    let path = dir.path().join("dense_attribute_rewrite.h5");
    {
        let file = create_dense_file(&path, file_space);
        let group = file
            .create_group("a")
            .expect("create group with dense attributes");
        add_dense_attributes(&group, &names, 1);
        add_dense_attribute_dataset(&file, &names, 1);
        file.close().unwrap();
    }
    let old = heap_allocations(
        &path,
        1,
        dense_heaps(&path)
            .into_iter()
            .filter(|heap| heap.owner == owner),
    );
    assert_eq!(
        old.iter()
            .filter(|&&(_, _, class)| class == PageClass::RawData)
            .count(),
        1
    );

    {
        let session = File::open_rw(&path).unwrap();
        if owner == "a" {
            session
                .group("a")
                .unwrap()
                .set_attr(&names.attributes[0], AttrValue::I64(99))
                .unwrap();
        } else {
            session
                .dataset("d")
                .unwrap()
                .set_attr(&names.attributes[0], AttrValue::I64(99))
                .unwrap();
        }
        session.commit().unwrap();
    }

    assert_released(file_space.page_size(), &released_regions(&path), &old);
    if let Some(page_size) = file_space.page_size() {
        assert_page_classes(&path, page_size, &old);
    }
    let c = hdf5::File::open(&path).unwrap();
    let location: hdf5::Location = if owner == "a" {
        (*c.group("a").unwrap()).clone()
    } else {
        (**c.dataset("d").unwrap()).clone()
    };
    let mut attr_names = location.attr_names().unwrap();
    attr_names.sort();
    let mut expected: Vec<String> = names
        .attributes
        .iter()
        .cloned()
        .chain([huge_name("huge_attr_", 0)])
        .collect();
    expected.sort();
    assert_eq!(attr_names, expected);
    let values: Vec<i64> = names
        .attributes
        .iter()
        .map(|attr_name| {
            location
                .attr(attr_name)
                .unwrap()
                .read_scalar::<i64>()
                .unwrap()
        })
        .collect();
    assert_eq!(values, [99, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(c.free_space(), free_bytes(&persistent_free(&path)));
}

#[rstest]
fn a_surviving_hard_link_keeps_a_dense_heap_live(#[values(false, true)] managed_fsm: bool) {
    let dir = tempdir().unwrap();
    let names = if managed_fsm {
        DenseNames::new()
    } else {
        exactly_filling_names(dir.path(), 1)
    };
    let path = dir.path().join("dense_heap_surviving_alias.h5");
    write_dense_fixture(&path, FileSpace::Managers, &names, 1, true);
    let allocations = owned_heap_allocations(&path, 1, managed_fsm);

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g").unwrap();
        session.commit().unwrap();
    }
    let free = persistent_free(&path);
    for &(addr, len, _) in &allocations {
        assert!(
            !overlaps_free(&free, addr, len),
            "a surviving hard link keeps the heap allocation ({addr}, {len}) live"
        );
    }
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset(&format!("alias/{}", names.links[0]))
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![0]
    );

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("alias").unwrap();
        session.root().delete("d").unwrap();
        session.commit().unwrap();
    }
    assert_released(None, &released_regions(&path), &allocations);
    assert_c_reads_after_reclaim(&path);
}

// libhdf5 1.10.11 allocates the persisted free-space managers of a paged file from the end of the
// file at every close (`H5MF_settle_meta_data_fsm` in `H5MF.c`), and the rest of their pages stays
// untracked, so each of its sessions grows the file.
#[cfg(feature = "__hdf5-1.12")]
#[rstest]
fn repeated_dense_storage_create_and_delete_on_a_paged_file_settles_at_a_steady_size(
    #[values(false, true)] managed_fsm: bool,
) {
    let dir = tempdir().unwrap();
    let names = if managed_fsm {
        DenseNames::new()
    } else {
        exactly_filling_names(dir.path(), 1)
    };
    let path = dir.path().join("dense_churn.h5");
    {
        let file = create_dense_file(&path, FileSpace::Pages(DENSE_PAGE_SIZE));
        file.new_dataset::<i32>()
            .shape([4])
            .create("keep")
            .unwrap()
            .write(&[1, 2, 3, 4])
            .unwrap();
        file.close().unwrap();
    }

    let sizes: Vec<(u64, u64)> = (0..8)
        .map(|_| {
            {
                let file = hdf5::File::open_rw(&path).unwrap();
                add_dense_objects(&file, &names, 1);
                file.close().unwrap();
            }
            let written = std::fs::metadata(&path).unwrap().len();
            let allocations = owned_heap_allocations(&path, 1, managed_fsm);
            {
                let session = File::open_rw(&path).unwrap();
                session.root().delete("g").unwrap();
                session.root().delete("d").unwrap();
                session.commit().unwrap();
            }
            assert_released(
                Some(DENSE_PAGE_SIZE),
                &released_regions(&path),
                &allocations,
            );
            assert_page_classes(&path, DENSE_PAGE_SIZE, &allocations);
            (written, std::fs::metadata(&path).unwrap().len())
        })
        .collect();

    assert_eq!(sizes, vec![sizes[0]; sizes.len()]);
    assert_c_reads_after_reclaim(&path);
}

const DENSE_PAGE_SIZE: u64 = 16 * 1024;
// `H5F_FILE_SPACE_PAGE_SIZE_MIN` in `H5Fprivate.h`, libhdf5 2.2.0.
const SMALL_PAGE_SIZE: u64 = 512;
const USERBLOCK: u64 = 512;
// `H5G_FHEAP_MAN_START_BLOCK_SIZE` in `H5Gdense.c` and `H5O_FHEAP_MAN_START_BLOCK_SIZE` in
// `H5Oprivate.h`, libhdf5 2.2.0.
const LINK_HEAP_START_BLOCK_SIZE: u64 = 512;
const ATTRIBUTE_HEAP_START_BLOCK_SIZE: u64 = 1024;
const HEAP_METADATA_SIGNATURES: [&[u8; 4]; 8] = [
    fractal_heap::SIGNATURE,
    fractal_heap::DIRECT_BLOCK_SIGNATURE,
    fractal_heap::INDIRECT_BLOCK_SIGNATURE,
    btree_v2::SIGNATURE,
    btree_v2::INTERNAL_SIGNATURE,
    btree_v2::LEAF_SIGNATURE,
    free_space::SIGNATURE,
    free_space::SECTIONS_SIGNATURE,
];
// The small-sized managers for `H5FD_MEM_SUPER` and `H5FD_MEM_DRAW`, first and third in a version
// 1 File Space Info message, defined in "The File Space Info Message" of the format
// specification, version 4.0.
const SMALL_METADATA_MANAGER: usize = 0;
const SMALL_RAW_MANAGER: usize = 2;
