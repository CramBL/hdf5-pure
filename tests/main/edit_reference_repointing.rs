//! An object reference the file already stores keeps naming its target across a
//! commit that moves that target's object header (issue #324).
//!
//! A `File::open_rw` commit rebuilds a dirty object's header at a fresh address
//! and repoints the one *link* that names it. An object reference is an address
//! stored as data, and nothing about rewriting a header updates one — so before
//! this, a reference dataset the commit never touched was left naming the header
//! the same commit freed. The reads that follow are the point of these tests:
//! the file stayed *readable*, resolving to a stale copy of the object, and only
//! became unreadable once something reused the span.

use hdf5_pure::{
    AttrValue, CompoundTypeBuilder, Datatype, DatatypeByteOrder, File, FileBuilder,
    FileSpaceStrategy, FixedPointLayout, Group, Object, ReferenceType,
};
use tempfile::tempdir;
use test_util_hdf5::dataset::Unlimited;

/// The names of the datasets the file's first stored reference resolves to,
/// dereferenced through the reference rather than looked up by path.
///
/// Going through `dereference` is what makes these tests about #324 at all: the
/// link graph was always right, and reading `g` by *path* passed throughout.
fn referenced_group_members(path: &std::path::Path, dataset: &str) -> Vec<String> {
    let file = File::open(path).unwrap();
    let objects = file.dataset(dataset).unwrap().dereference().unwrap();
    match &objects[0] {
        Object::Group(g) => sorted_datasets(g),
        other => panic!("expected the reference to resolve to a group, got {other:?}"),
    }
}

fn sorted_datasets(g: &Group) -> Vec<String> {
    let mut names = g.datasets().unwrap();
    names.sort();
    names
}

/// A file holding `g` (with one dataset) and a `refs` dataset naming `g`.
fn build_file_referencing_a_group(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    b.create_dataset("refs").with_path_references(&["g"]);
    b.write(path).unwrap();
}

/// Commit `count` unrelated small datasets, one per commit, so the space the
/// earlier commits vacated is reused. A stale reference reads plausible bytes
/// until that happens, which is why every test that wants to show the reference
/// is *sound* rather than merely lucky runs this first.
fn churn(path: &std::path::Path, count: i32) {
    for i in 0..count {
        let session = File::open_rw(path).unwrap();
        session
            .root()
            .create_dataset(&format!("churn{i}"), |b| {
                b.with_i32_data(&[i]);
            })
            .unwrap();
        session.commit().unwrap();
    }
}

#[test]
fn a_reference_to_a_group_follows_it_when_the_group_gains_a_child() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("group_target.h5");
    build_file_referencing_a_group(&path);

    // Adding a child dirties `g`, so its header is rebuilt at a fresh address.
    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("g/extra", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();
    session.commit().unwrap();
    drop(session);

    assert_eq!(
        referenced_group_members(&path, "refs"),
        vec!["extra".to_string(), "inner".to_string()],
        "the reference must resolve to the group as the commit left it, not to \
         the superseded copy that is missing `extra`"
    );

    // And it stays right once later commits have spent the freed space.
    churn(&path, 10);
    assert_eq!(
        referenced_group_members(&path, "refs"),
        vec!["extra".to_string(), "inner".to_string()],
    );
}

#[test]
fn a_reference_to_a_dataset_follows_its_relocating_attribute_edit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dataset_target.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("target").with_i32_data(&[1, 2, 3]);
    b.create_dataset("refs").with_path_references(&["target"]);
    b.write(&path).unwrap();

    // An attribute edit rewrites the dataset's header elsewhere.
    let session = File::open_rw(&path).unwrap();
    session
        .dataset("target")
        .unwrap()
        .set_attr("units", AttrValue::I32(1))
        .unwrap();
    session.commit().unwrap();
    drop(session);
    churn(&path, 10);

    let file = File::open(&path).unwrap();
    let objects = file.dataset("refs").unwrap().dereference().unwrap();
    match &objects[0] {
        Object::Dataset(d) => {
            assert_eq!(d.read_i32().unwrap(), vec![1, 2, 3]);
            assert_eq!(
                d.attrs().unwrap().keys().collect::<Vec<_>>(),
                vec!["units"],
                "the reference must resolve to the edited header, attribute and all"
            );
        }
        other => panic!("expected a dataset, got {other:?}"),
    }
}

#[test]
fn a_reference_to_the_root_group_follows_the_rebuilt_root() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("root_target.h5");

    // Every commit rebuilds the root group, so a reference to it is the one that
    // moves on *every* commit rather than only on the ones that touch its target.
    let mut b = FileBuilder::new();
    b.create_dataset("alpha").with_i32_data(&[1]);
    b.create_dataset("refs").with_path_references(&[""]);
    b.write(&path).unwrap();

    churn(&path, 5);

    let file = File::open(&path).unwrap();
    let objects = file.dataset("refs").unwrap().dereference().unwrap();
    match &objects[0] {
        Object::Group(g) => {
            let names = sorted_datasets(g);
            assert!(
                names.contains(&"alpha".to_string()) && names.contains(&"churn4".to_string()),
                "the reference must resolve to the current root, which holds every \
                 committed dataset; got {names:?}"
            );
        }
        other => panic!("expected the root group, got {other:?}"),
    }
}

#[test]
fn every_element_of_a_multi_reference_dataset_is_repointed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("many.h5");

    let mut b = FileBuilder::new();
    for name in ["a", "b", "c"] {
        let mut g = b.create_group(name);
        g.create_dataset("inner").with_i32_data(&[1]);
        b.add_group(g.finish());
    }
    b.create_dataset("refs")
        .with_path_references(&["a", "b", "c"]);
    b.write(&path).unwrap();

    // One commit dirties all three groups, so all three headers move at once.
    let session = File::open_rw(&path).unwrap();
    for name in ["a", "b", "c"] {
        session
            .root()
            .create_dataset(&format!("{name}/extra"), |b| {
                b.with_i32_data(&[9]);
            })
            .unwrap();
    }
    session.commit().unwrap();
    drop(session);
    churn(&path, 10);

    let file = File::open(&path).unwrap();
    let objects = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(objects.len(), 3);
    for object in &objects {
        match object {
            Object::Group(g) => assert_eq!(
                sorted_datasets(g),
                vec!["extra".to_string(), "inner".to_string()]
            ),
            other => panic!("expected a group, got {other:?}"),
        }
    }
}

#[test]
fn a_reference_inside_a_compound_element_is_repointed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("compound.h5");

    // A compound of {object reference, i32} is the shape the reference C library
    // uses for a dimension scale's `REFERENCE_LIST`, and the one that proves the
    // walk addresses a reference *inside* an element rather than only an element
    // that is one.
    let mut b = FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    b.create_dataset("plain").with_path_references(&["g"]);
    b.write(&path).unwrap();

    // Read the resolved address, then restage it inside a compound element.
    let address = {
        let file = File::open(&path).unwrap();
        let raw = file.dataset("plain").unwrap().read_raw().unwrap();
        u64::from_le_bytes(raw[..8].try_into().unwrap())
    };
    let mut element = address.to_le_bytes().to_vec();
    element.extend_from_slice(&7i32.to_le_bytes());
    let compound = CompoundTypeBuilder::with_size(12)
        .field(
            "target",
            0,
            Datatype::Reference {
                size: 8,
                ref_type: ReferenceType::Object,
            },
        )
        .field(
            "index",
            8,
            Datatype::FixedPoint {
                size: 4,
                byte_order: DatatypeByteOrder::LittleEndian,
                layout: FixedPointLayout {
                    signed: true,
                    bit_offset: 0,
                    bit_precision: 32,
                },
            },
        )
        .build()
        .unwrap();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("pairs", |b| {
                b.with_raw_data(compound, element, 1);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("g/extra", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();
    session.commit().unwrap();
    drop(session);
    churn(&path, 10);

    // `dereference` reads a dataset whose *element* is a reference, so the
    // compound is checked against the plain reference dataset beside it: both
    // name `g`, so after the commit both must hold the same address, and the
    // plain one proves that address is the group as the commit left it.
    assert_eq!(
        referenced_group_members(&path, "plain"),
        vec!["extra".to_string(), "inner".to_string()],
    );
    let file = File::open(&path).unwrap();
    let plain = file.dataset("plain").unwrap().read_raw().unwrap();
    let pairs = file.dataset("pairs").unwrap().read_raw().unwrap();
    assert_eq!(
        pairs[..8],
        plain[..8],
        "the reference in the compound's first member must be repointed too"
    );
    assert_ne!(
        pairs[..8],
        address.to_le_bytes(),
        "and it must actually have moved, or this test proves nothing"
    );
    // The member beside it is untouched: the patch is eight bytes, not an element.
    assert_eq!(i32::from_le_bytes(pairs[8..12].try_into().unwrap()), 7);
}

#[test]
fn a_reference_to_a_deleted_object_is_left_dangling() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("deleted.h5");
    build_file_referencing_a_group(&path);

    let before = {
        let file = File::open(&path).unwrap();
        file.dataset("refs").unwrap().read_raw().unwrap()
    };

    // Deleting the target is not a relocation: the object is gone, and HDF5 has
    // no rule that says a reference to it becomes anything else. The address is
    // left exactly as it was, which is also what the reference C library does.
    let session = File::open_rw(&path).unwrap();
    session.root().delete("g").unwrap();
    session.commit().unwrap();
    drop(session);

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("refs").unwrap().read_raw().unwrap(),
        before,
        "a reference whose target was deleted must be left alone, not redirected"
    );
}

#[test]
fn references_are_repointed_on_a_userblock_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("userblock.h5");

    // Stored addresses are relative to the superblock base, so a userblock is
    // where an off-by-`base` in either direction shows up.
    let mut b = FileBuilder::new();
    b.with_userblock(1024);
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    b.create_dataset("refs").with_path_references(&["g"]);
    b.write(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("g/extra", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();
    session.commit().unwrap();
    drop(session);
    churn(&path, 10);

    assert_eq!(
        referenced_group_members(&path, "refs"),
        vec!["extra".to_string(), "inner".to_string()],
    );
}

/// The same edit on each of the three commit tails: the plain one, the one that
/// persists its free space, and the paged one. Each repoints the superblock in
/// its own code, so each needs its own proof that the fixup runs after it.
fn reference_survives_on(path: &std::path::Path, strategy: Option<FileSpaceStrategy>) {
    let mut b = FileBuilder::new();
    if let Some(strategy) = strategy {
        b.with_file_space_strategy(strategy, true, 1);
    }
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    b.create_dataset("refs").with_path_references(&["g"]);
    b.write(path).unwrap();

    let session = File::open_rw(path).unwrap();
    session
        .root()
        .create_dataset("g/extra", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();
    session.commit().unwrap();
    drop(session);
    churn(path, 10);

    assert_eq!(
        referenced_group_members(path, "refs"),
        vec!["extra".to_string(), "inner".to_string()],
    );
}

#[test]
fn references_are_repointed_on_a_persisting_file() {
    let dir = tempdir().unwrap();
    reference_survives_on(
        &dir.path().join("persist.h5"),
        Some(FileSpaceStrategy::FsmAggr),
    );
}

#[test]
fn references_are_repointed_on_a_paged_file() {
    let dir = tempdir().unwrap();
    reference_survives_on(&dir.path().join("paged.h5"), Some(FileSpaceStrategy::Page));
}

#[test]
fn repeated_commits_in_one_session_each_repoint() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("many_commits.h5");
    build_file_referencing_a_group(&path);

    // One session, several commits, each moving `g` again. The session caches
    // "this file holds no references" to skip the walk, and this is the file
    // where that cache must never be set.
    let session = File::open_rw(&path).unwrap();
    for i in 0..5 {
        session
            .root()
            .create_dataset(&format!("g/d{i}"), |b| {
                b.with_i32_data(&[i]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    drop(session);
    churn(&path, 10);

    let mut expected: Vec<String> = (0..5).map(|i| format!("d{i}")).collect();
    expected.push("inner".to_string());
    expected.sort();
    assert_eq!(referenced_group_members(&path, "refs"), expected);
}

#[test]
fn a_reference_dataset_added_after_a_reference_free_walk_is_still_repointed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("added_later.h5");

    // The file starts with nothing a reference could live in, so the first
    // commit's walk proves it reference-free and licenses every later commit to
    // skip the walk. Adding a reference dataset has to retire that proof.
    let mut b = FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    b.write(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("plain", |b| {
            b.with_i32_data(&[1]);
        })
        .unwrap();
    session.commit().unwrap(); // the walk that finds no references
    session
        .root()
        .create_dataset("refs", |b| {
            b.with_path_references(&["g"]);
        })
        .unwrap();
    session.commit().unwrap(); // must retire the proof
    session
        .root()
        .create_dataset("g/extra", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();
    session.commit().unwrap(); // and this one must walk
    drop(session);
    churn(&path, 10);

    assert_eq!(
        referenced_group_members(&path, "refs"),
        vec!["extra".to_string(), "inner".to_string()],
    );
}

#[test]
fn a_copy_made_in_the_same_commit_that_moves_its_target_is_repointed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("copied.h5");
    build_file_referencing_a_group(&path);

    // An in-file copy re-emits its source's element bytes verbatim, so the copy
    // is written naming the address the source named — which this same commit is
    // vacating. The copy is not refused for that (the reference it repeats is
    // one the file already made), and the walk that follows the commit sees the
    // copy as part of the committed tree and repoints it along with its source.
    let session = File::open_rw(&path).unwrap();
    session.copy("refs", "refs_copy").unwrap();
    session
        .root()
        .create_dataset("g/extra", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();
    session.commit().unwrap();
    drop(session);
    churn(&path, 10);

    for dataset in ["refs", "refs_copy"] {
        assert_eq!(
            referenced_group_members(&path, dataset),
            vec!["extra".to_string(), "inner".to_string()],
            "{dataset} must name the group as the commit left it"
        );
    }
}

#[test]
fn a_reference_to_an_appended_dataset_follows_its_relocating_append() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("appended.h5");

    // A staged append rebuilds the dataset's chunk index and rewrites its header
    // elsewhere, which moves the object a reference names just as an attribute
    // edit does — and it arrives through a different staged collection.
    let mut b = FileBuilder::new();
    Unlimited::new("target", &[1i32, 2, 3], 3).add_to(&mut b);
    b.create_dataset("refs").with_path_references(&["target"]);
    b.write(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    session
        .dataset("target")
        .unwrap()
        .append_staged(|a| {
            a.append_i32(&[4, 5, 6]);
        })
        .unwrap();
    session.commit().unwrap();
    drop(session);
    churn(&path, 10);

    let file = File::open(&path).unwrap();
    let objects = file.dataset("refs").unwrap().dereference().unwrap();
    match &objects[0] {
        Object::Dataset(d) => assert_eq!(d.read_i32().unwrap(), vec![1, 2, 3, 4, 5, 6]),
        other => panic!("expected the appended dataset, got {other:?}"),
    }
}
