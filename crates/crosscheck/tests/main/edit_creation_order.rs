#![cfg(feature = "__hdf5-1.10")]
//! In-place edits of objects that track attribute creation order (issue #416).
//!
//! `H5Pset_attr_creation_order` — h5py's `track_order=True`, and what netCDF-4
//! sets on every object it writes — makes every object-header message record six
//! bytes wide instead of four, carrying a creation index. `EditSession` used to
//! refuse such an object outright.
//!
//! The fixtures have to come from the reference C library: nothing in this
//! crate's whole-file writer emits creation order, so a fixture written here
//! would only test the editor against itself. And the interesting assertion is
//! not "the attributes are still there" — this crate's own reader ignores
//! creation order entirely, and would say yes to a file whose indexes are
//! nonsense. It is what the C library makes of the order afterwards:
//! `H5Aiterate2` over `H5_INDEX_CRT_ORDER` (h5py's `track_order` iteration) and
//! `H5Aget_info_by_name`'s `corder`, which name the exact index each attribute
//! carries.
//!
//! The tail of the file covers the same ground for **link** creation order, the
//! separate mechanism `H5Pset_link_creation_order` turns on: there the judges
//! are `H5Literate2` over `H5_INDEX_CRT_ORDER`, `H5Lget_info2`'s `corder`, and
//! `H5Gget_info`'s `max_corder`.

use std::path::Path;

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5::plist::group_create::{
    AttrCreationOrder, GroupCreate, GroupCreateBuilder, LinkCreationOrder,
};
use hdf5::{IndexType, IterationOrder, LinkInfo};
use hdf5_pure::{AttrValue, Error, File, FormatError};
use tempfile::tempdir;
use test_util_hdf5::absence;

/// Whether the object creation property list should also index creation order,
/// which is what adds the creation-order B-tree once attributes go dense.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Indexed {
    No,
    Yes,
}

impl Indexed {
    /// The attribute creation-order property: tracked under either variant, and
    /// indexed under [`Indexed::Yes`].
    fn attr_order(self) -> AttrCreationOrder {
        match self {
            Self::No => AttrCreationOrder::Tracked,
            Self::Yes => AttrCreationOrder::Indexed,
        }
    }

    /// The link creation-order property: tracked under either variant, and
    /// indexed under [`Indexed::Yes`].
    fn link_order(self) -> LinkCreationOrder {
        match self {
            Self::No => LinkCreationOrder::Tracked,
            Self::Yes => LinkCreationOrder::Indexed,
        }
    }
}

/// A group creation property list tracking attribute creation order, and link
/// creation order too when `links`.
fn tracking_gcpl(indexed: Indexed, links: bool) -> GroupCreate {
    let mut builder = GroupCreateBuilder::new();
    builder.attr_creation_order(indexed.attr_order());
    if links {
        builder.link_creation_order(indexed.link_order());
    }
    builder.finish().expect("a group creation property list")
}

/// Write a file whose group `/g` and dataset `/d` both track attribute creation
/// order, each carrying one integer attribute per name in `names`, created in
/// that order and valued by its position.
///
/// The file creation property list carries the same setting so the root group
/// tracks it too, which is what h5py's `File(..., track_order=True)` does.
fn write_tracked(path: &Path, names: &[String], indexed: Indexed) {
    write_tracked_with_persistence(path, names, indexed, false);
}

/// Writes the [`write_tracked`] fixture with an optional persistent free-space manager.
fn write_tracked_with_persistence(path: &Path, names: &[String], indexed: Indexed, persist: bool) {
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
        .set_gcpl(&tracking_gcpl(indexed, false))
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

#[test]
fn c_library_accepts_persisted_reclaimed_dense_attribute_indexes() {
    let dir = tempdir().unwrap();
    let attribute_names = names(12);

    for indexed in [Indexed::No, Indexed::Yes] {
        let suffix = if indexed == Indexed::Yes {
            "indexed"
        } else {
            "name_only"
        };
        let path = dir
            .path()
            .join(format!("dense_attribute_reclaim_{suffix}.h5"));
        write_tracked_with_persistence(&path, &attribute_names, indexed, true);

        if indexed == Indexed::Yes {
            assert_eq!(
                Owner::open(&path, "d").name_by_creation_index(0),
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
        assert_eq!(Owner::open(&path, "g").names(IndexType::Name).len(), 12);
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
/// `Indexed::No` uses ordinary name indexing, while `Indexed::Yes` also tracks and indexes link
/// creation order. `/g/shared` has a root hard link so deleting the group exercises child lifetime
/// independently of link-index ownership.
fn dense_link_gcpl(indexed: Indexed) -> GroupCreate {
    let mut builder = GroupCreateBuilder::new();
    if indexed == Indexed::Yes {
        builder.link_creation_order(Indexed::Yes.link_order());
    }
    builder
        .finish()
        .expect("a dense-link group creation property list")
}

fn write_dense_link_reclaim_fixture(path: &Path, indexed: Indexed, paged: bool, alias_group: bool) {
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
        (Indexed::No, false),
        (Indexed::Yes, false),
        (Indexed::No, true),
        (Indexed::Yes, true),
    ] {
        let path = dir.path().join(format!(
            "dense_group_reclaim_{}_{}.h5",
            if indexed == Indexed::Yes {
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
            indexed == Indexed::Yes
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
    write_dense_link_reclaim_fixture(&path, Indexed::Yes, false, true);
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

/// A file whose group `/g` tracks *link* creation order, as netCDF-4 writes,
/// holding one dataset per name in `links`, created in that order.
fn write_link_tracked(path: &Path, links: &[&str]) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::latest()))
        .with_fcpl(|p| {
            p.attr_creation_order(Indexed::Yes.attr_order())
                .link_creation_order(Indexed::Yes.link_order())
        })
        .create(path)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    let group = file
        .create_group_builder()
        .set_gcpl(&tracking_gcpl(Indexed::Yes, true))
        .create("g")
        .expect("create group");
    for link in links {
        group
            .new_dataset::<i32>()
            .shape([4])
            .create(*link)
            .unwrap_or_else(|e| panic!("create dataset {link}: {e}"));
    }
    file.close().unwrap();
}

/// The links of `group` in `path`, each with the info the C library reports for
/// it, ordered by `index_type`.
fn links_of(path: &Path, group: &str, index_type: IndexType) -> Vec<(String, LinkInfo)> {
    let file = hdf5::File::open(path)
        .unwrap_or_else(|e| panic!("the C library opens {}: {e}", path.display()));
    file.group(group)
        .unwrap_or_else(|e| panic!("the C library opens /{group}: {e}"))
        .links(index_type, IterationOrder::Increasing)
        .unwrap_or_else(|e| panic!("the C library iterates the links of /{group}: {e}"))
}

/// An open handle to `/g` or `/d`, whichever `path` names.
struct Owner {
    _file: hdf5::File,
    location: hdf5::Location,
}

impl Owner {
    fn open(path: &Path, object: &str) -> Self {
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
        Self {
            _file: file,
            location,
        }
    }

    /// Attribute names ordered by `index_type`.
    fn names(&self, index_type: IndexType) -> Vec<String> {
        self.location
            .attr_names_by(index_type, IterationOrder::Increasing)
            .unwrap_or_else(|e| panic!("the C library iterates attributes: {e}"))
    }

    /// The name of the `n`th attribute in creation order, read through
    /// `H5Aopen_by_idx`.
    ///
    /// Distinct from [`names`](Self::names) on purpose: `H5Aiterate2` with an
    /// increasing order builds its table from the *name* index whatever the
    /// requested index type, so it reads the creation index out of the name
    /// index's records and never touches the creation-order B-tree. This walks
    /// `H5A__dense_open_by_idx`, which indexes straight into that B-tree
    /// whenever the Attribute Info message names one — so a dense object whose
    /// message declares the index has to actually carry it.
    fn name_by_creation_index(&self, n: u64) -> String {
        self.location
            .attr_by_index(IndexType::CreationOrder, IterationOrder::Increasing, n)
            .unwrap_or_else(|e| panic!("the C library opens attribute {n} by creation order: {e}"))
            .name()
    }

    /// The value of the integer attribute `name`.
    fn value(&self, name: &str) -> i32 {
        self.location
            .attr(name)
            .unwrap_or_else(|e| panic!("the C library opens attribute {name}: {e}"))
            .read_scalar::<i32>()
            .unwrap_or_else(|e| panic!("the C library reads attribute {name}: {e}"))
    }

    /// The creation index the C library reports for the attribute `name`.
    fn creation_index(&self, name: &str) -> u32 {
        self.location
            .attr_info(name)
            .unwrap_or_else(|e| panic!("the C library reads info for attribute {name}: {e}"))
            .creation_order
            .unwrap_or_else(|| {
                panic!("{name} carries no creation index, so the object stopped tracking the order")
            })
    }
}

/// [`links_in_creation_order_of`] for `/g`, the group every fixture here tracks.
fn links_in_creation_order(path: &Path) -> (Vec<String>, i64) {
    links_in_creation_order_of(path, "g")
}

/// The names of `group`'s links in link creation order, and the highest creation
/// index the group has ever assigned — the counter a deletion must leave alone.
fn links_in_creation_order_of(path: &Path, group: &str) -> (Vec<String>, i64) {
    let names = links_of(path, group, IndexType::CreationOrder)
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    let file = hdf5::File::open(path)
        .unwrap_or_else(|e| panic!("the C library opens {}: {e}", path.display()));
    let info = file
        .group(group)
        .unwrap_or_else(|e| panic!("the C library opens /{group}: {e}"))
        .info()
        .unwrap_or_else(|e| panic!("the C library reads info for /{group}: {e}"));
    (names, info.max_corder)
}

/// [`link_creation_index_of`] for `/g`, the group every fixture here tracks.
fn link_creation_index(path: &Path, link: &str) -> i64 {
    link_creation_index_of(path, "g", link)
}

/// The creation index the C library reports for the link `<group>/<link>`, which
/// must be one the group actually records: a link written without one reads back
/// with `corder_valid` clear, and the assertion below is what catches that.
fn link_creation_index_of(path: &Path, group: &str, link: &str) -> i64 {
    let (_, info) = links_of(path, group, IndexType::Name)
        .into_iter()
        .find(|(name, _)| name == link)
        .unwrap_or_else(|| panic!("/{group}/{link} is not a link the C library lists"));
    info.creation_order.unwrap_or_else(|| {
        panic!(
            "/{group}/{link} carries no creation index, so it is unnumbered in the group's \
             link order"
        )
    })
}

fn names(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("a{i:02}")).collect()
}

/// Both objects of a fixture, so every test covers a group and a dataset — the
/// two headers the editor rebuilds by different routes.
const OBJECTS: [&str; 2] = ["g", "d"];

/// Set `name` on both `/g` and `/d` through one edit session.
fn set_on_both(path: &Path, name: &str, value: i64) {
    let s = File::open_rw(path).expect("the editor opens a tracked file");
    s.group("g")
        .expect("the group is reachable")
        .set_attr(name, AttrValue::I64(value))
        .expect("the group takes an attribute");
    s.dataset("d")
        .expect("the dataset is reachable")
        .set_attr(name, AttrValue::I64(value))
        .expect("the dataset takes an attribute");
    s.commit().expect("the commit lands");
}

#[test]
fn an_object_tracking_creation_order_can_be_edited_at_all() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(3), Indexed::Yes);

    set_on_both(&p, "added", 99);

    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(IndexType::Name),
            ["a00", "a01", "a02", "added"],
            "/{object} lost an attribute by name",
        );
        // The new attribute takes the next creation index, so it iterates last —
        // which is exactly what h5py's `track_order` iteration shows.
        assert_eq!(
            owner.names(IndexType::CreationOrder),
            ["a00", "a01", "a02", "added"],
            "/{object} did not put the new attribute last in creation order",
        );
        assert_eq!(owner.value("added"), 99);
        assert_eq!(owner.value("a01"), 1, "/{object} lost an existing value");
        assert_eq!(owner.creation_index("added"), 3);
    }
}

#[test]
fn the_editor_reads_a_tracked_object_back_itself() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(3), Indexed::Yes);
    set_on_both(&p, "added", 7);

    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), 4);
    assert_eq!(attrs.get("added"), Some(&AttrValue::I64(7)));
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap().len(), 4);
}

#[test]
fn overwriting_an_attribute_keeps_the_creation_index_it_had() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(3), Indexed::Yes);

    set_on_both(&p, "a00", -1);

    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(IndexType::CreationOrder),
            ["a00", "a01", "a02"],
            "/{object} moved an overwritten attribute in the creation order",
        );
        assert_eq!(owner.value("a00"), -1);
        assert_eq!(owner.creation_index("a00"), 0);
    }
}

#[test]
fn deleting_an_attribute_leaves_a_gap_rather_than_renumbering() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(4), Indexed::Yes);

    // One from the middle and one from the end: the middle deletion is what
    // leaves a gap in the surviving indexes, and the end one is what separates
    // the counter the Attribute Info message records from the highest index
    // still in use.
    {
        let s = File::open_rw(&p).unwrap();
        let g = s.group("g").unwrap();
        let mut d = s.dataset("d").unwrap();
        for name in ["a01", "a03"] {
            g.remove_attr(name).unwrap();
            d.remove_attr(name).unwrap();
        }
        s.commit().unwrap();
    }
    // A fresh attribute takes index 4, neither a freed index nor one past the
    // highest survivor: the reference C library hands them out from a counter
    // that only rises.
    set_on_both(&p, "added", 5);

    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(IndexType::CreationOrder),
            ["a00", "a02", "added"]
        );
        assert_eq!(owner.creation_index("a00"), 0);
        assert_eq!(
            owner.creation_index("a02"),
            2,
            "/{object} renumbered the attributes that survived",
        );
        assert_eq!(
            owner.creation_index("added"),
            4,
            "/{object} lowered the creation-index counter a deletion must leave alone",
        );
    }
}

#[test]
fn a_compact_set_crossing_the_threshold_carries_its_creation_order_into_the_heap() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // Eight compact attributes; one is deleted and six more added in one
    // session. That is past the writer's compact threshold, so the whole set is
    // rebuilt into a fractal heap — and the deletion is what makes each
    // attribute's creation index differ from its position in the rebuilt set,
    // so an index taken from the position would show up here.
    write_tracked(&p, &names(8), Indexed::Yes);

    {
        let s = File::open_rw(&p).unwrap();
        let g = s.group("g").unwrap();
        let mut d = s.dataset("d").unwrap();
        g.remove_attr("a01").unwrap();
        d.remove_attr("a01").unwrap();
        for i in 8..14 {
            g.set_attr(&format!("a{i:02}"), AttrValue::I64(i as i64))
                .unwrap();
            d.set_attr(&format!("a{i:02}"), AttrValue::I64(i as i64))
                .unwrap();
        }
        s.commit().unwrap();
    }

    let expected: Vec<String> = names(14).into_iter().filter(|n| n != "a01").collect();
    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(IndexType::CreationOrder),
            expected,
            "/{object} lost the creation order on the way into the heap",
        );
        assert_eq!(owner.names(IndexType::Name), expected);
        // Straight through the creation-order B-tree, which this object's
        // Attribute Info message declares.
        for (n, name) in expected.iter().enumerate() {
            assert_eq!(
                &owner.name_by_creation_index(n as u64),
                name,
                "/{object} does not carry the creation-order index it declares",
            );
        }
        for name in &expected {
            let created: u32 = name[1..].parse().expect("a fixture name is a index");
            assert_eq!(owner.value(name), created as i32, "/{object} {name}");
            assert_eq!(
                owner.creation_index(name),
                created,
                "/{object} gave {name} an index that is not the order it was created in",
            );
        }
    }
}

#[test]
fn a_tracked_set_goes_dense_without_a_creation_order_index_when_the_object_has_none() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // Tracked but not *indexed*: the heap gets a name index only, and the
    // creation index each attribute carries lives in that index's records.
    write_tracked(&p, &names(8), Indexed::No);

    {
        let s = File::open_rw(&p).unwrap();
        let mut d = s.dataset("d").unwrap();
        for i in 8..13 {
            d.set_attr(&format!("a{i:02}"), AttrValue::I64(i as i64))
                .unwrap();
        }
        s.commit().unwrap();
    }

    let owner = Owner::open(&p, "d");
    assert_eq!(owner.names(IndexType::Name), names(13));
    for (i, name) in names(13).iter().enumerate() {
        assert_eq!(owner.creation_index(name), i as u32);
    }
}

#[test]
fn an_object_already_dense_keeps_its_creation_order_across_an_edit() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // Twelve attributes is past the C library's own compact threshold, so both
    // objects already store their attributes in a fractal heap.
    write_tracked(&p, &names(12), Indexed::Yes);

    set_on_both(&p, "added", 42);

    let mut expected: Vec<String> = names(12);
    expected.push("added".to_string());
    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(IndexType::CreationOrder),
            expected,
            "/{object} reordered a dense set that was already tracked",
        );
        assert_eq!(owner.value("added"), 42);
        for (n, name) in expected.iter().enumerate() {
            assert_eq!(
                &owner.name_by_creation_index(n as u64),
                name,
                "/{object} does not carry the creation-order index it declares",
            );
        }
        for (i, name) in names(12).iter().enumerate() {
            assert_eq!(owner.creation_index(name), i as u32);
        }
        assert_eq!(owner.creation_index("added"), 12);
    }
}

/// A group that tracks link creation order takes additions: each new link is
/// numbered from the running maximum its Link Info message records, and that
/// maximum is bumped by the number added.
///
/// The reference C library is the judge, because this crate's own reader ignores
/// link creation order entirely: `H5Literate2` over `H5_INDEX_CRT_ORDER` walks
/// the per-link creation indexes, `H5Lget_info2`'s `corder` names the index one
/// link carries, and `H5Gget_info`'s `max_corder` is the running counter.
#[test]
fn links_created_in_a_tracked_group_take_the_next_creation_indexes() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_link_tracked(&p, &["first", "second"]);

    // Its attributes are editable, as they were before this change.
    {
        let s = File::open_rw(&p).unwrap();
        s.group("g")
            .unwrap()
            .set_attr("note", AttrValue::I64(1))
            .unwrap();
        s.commit().unwrap();
    }
    assert_eq!(Owner::open(&p, "g").value("note"), 1);

    // A dataset and a child group, both added in one commit.
    {
        let s = File::open_rw(&p).unwrap();
        let g = s.group("g").unwrap();
        g.create_dataset("added_d", |d| {
            d.with_i32_data(&[1, 2]);
        })
        .unwrap();
        g.create_group("added_g").unwrap();
        s.commit().expect("a tracked group takes an addition");
    }

    let (names, max_corder) = links_in_creation_order(&p);
    assert_eq!(
        names,
        ["first", "second", "added_d", "added_g"],
        "the additions must follow the originals in link creation order",
    );
    assert_eq!(
        max_corder, 4,
        "the counter must advance by the number of links added",
    );
    // Two links added in one commit are numbered consecutively, in the order
    // they were placed.
    assert_eq!(link_creation_index(&p, "added_d"), 2);
    assert_eq!(link_creation_index(&p, "added_g"), 3);
    // The originals keep the indexes they were created with.
    assert_eq!(link_creation_index(&p, "first"), 0);
    assert_eq!(link_creation_index(&p, "second"), 1);
}

/// Deleting a link from a group that tracks link creation order leaves a gap in
/// the order and touches no counter, the same shape this crate already gives an
/// attribute deletion — and the next addition takes the index past the old
/// maximum rather than reusing the gap.
#[test]
fn a_tracked_group_leaves_a_deletion_gap_and_adds_past_it() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_link_tracked(&p, &["first", "middle", "last"]);
    assert_eq!(
        links_in_creation_order(&p),
        (
            vec![
                "first".to_string(),
                "middle".to_string(),
                "last".to_string()
            ],
            3
        ),
        "the fixture does not track link creation order",
    );

    {
        let s = File::open_rw(&p).unwrap();
        s.group("g").unwrap().delete("middle").unwrap();
        s.commit().expect("a deletion needs no creation index");
    }

    let (names, max_corder) = links_in_creation_order(&p);
    assert_eq!(
        names,
        ["first", "last"],
        "the survivors lost their creation order",
    );
    assert_eq!(
        max_corder, 3,
        "a deletion must not lower the group's link creation-index counter",
    );

    {
        let s = File::open_rw(&p).unwrap();
        s.group("g")
            .unwrap()
            .create_dataset("added", |d| {
                d.with_i32_data(&[1, 2]);
            })
            .unwrap();
        s.commit().expect("a tracked group takes an addition");
    }

    let (names, max_corder) = links_in_creation_order(&p);
    assert_eq!(names, ["first", "last", "added"]);
    assert_eq!(max_corder, 4);
    assert_eq!(
        link_creation_index(&p, "added"),
        3,
        "the addition must take the index past the old maximum, not the gap the \
         deletion left at 1",
    );
}

/// A group at the compact-storage threshold its Group Info message declares
/// cannot take another link: past it the reference C library moves the links
/// into a fractal heap indexed by a creation-order B-tree, which this crate does
/// not write (issue #102). The refusal lands before any byte changes.
#[test]
fn a_tracked_group_at_its_compact_threshold_is_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // Eight links is the C library's default maximum compact value, so the
    // fixture is the largest tracked group that still stores its links in the
    // object header.
    let links: Vec<String> = (0..8).map(|i| format!("d{i}")).collect();
    let refs: Vec<&str> = links.iter().map(String::as_str).collect();
    write_link_tracked(&p, &refs);
    assert_eq!(links_in_creation_order(&p).1, 8);
    let before = std::fs::read(&p).unwrap();

    // Scoped so the session — and the file lock it holds, which Windows
    // enforces — is gone before the bytes are read back.
    let err = {
        let s = File::open_rw(&p).unwrap();
        s.group("g")
            .unwrap()
            .create_dataset("extra", |d| {
                d.with_i32_data(&[1, 2]);
            })
            .unwrap();
        s.commit().unwrap_err()
    };
    assert!(
        matches!(&err, Error::EditUnsupported(m)
            if m.contains("link creation order") && m.contains("dense")),
        "got: {err}",
    );
    assert_eq!(
        std::fs::read(&p).unwrap(),
        before,
        "the refusal wrote bytes"
    );
}

/// A tracked group whose links are *already* dense is refused for the same
/// reason, and refused as it is read rather than as the addition is planned.
#[test]
fn a_dense_tracked_group_is_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // One past the default maximum compact value, so the C library wrote the
    // links into a fractal heap.
    let links: Vec<String> = (0..9).map(|i| format!("d{i}")).collect();
    let refs: Vec<&str> = links.iter().map(String::as_str).collect();
    write_link_tracked(&p, &refs);
    assert_eq!(links_in_creation_order(&p).1, 9);
    let before = std::fs::read(&p).unwrap();

    // Scoped so the session — and the file lock it holds, which Windows
    // enforces — is gone before the bytes are read back.
    let err = {
        let s = File::open_rw(&p).unwrap();
        s.group("g")
            .unwrap()
            .create_dataset("extra", |d| {
                d.with_i32_data(&[1, 2]);
            })
            .unwrap();
        s.commit().unwrap_err()
    };
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("dense")),
        "got: {err}",
    );
    assert_eq!(
        std::fs::read(&p).unwrap(),
        before,
        "the refusal wrote bytes"
    );
}

/// Copying a group that tracks link creation order reproduces that order: the
/// copy's Link Info message comes over verbatim, so each copied link has to keep
/// the creation index the source recorded for it or the copy would claim an
/// order none of its links carries.
#[test]
fn a_copied_group_keeps_its_link_creation_order() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_link_tracked(&p, &["first", "second", "third"]);

    {
        let s = File::open_rw(&p).unwrap();
        s.copy("g", "g2").unwrap();
        s.commit().expect("a tracked group is copyable");
    }

    assert_eq!(
        links_in_creation_order_of(&p, "g2"),
        (
            vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string()
            ],
            3
        ),
        "the copy lost the source's link creation order",
    );
    for (i, name) in ["first", "second", "third"].iter().enumerate() {
        assert_eq!(link_creation_index_of(&p, "g2", name), i as i64);
    }
}
