//! Appending through a handle reached by object reference (`Dataset::dereference`),
//! which has no resolvable path and so names its dataset by object-header address
//! (issue #198).

use hdf5_pure::{
    AttrValue, Error, File, FileAccessProperties, FileBuilder, MemoryStrategy, Object,
};

/// Open with the bounded engine demanded rather than merely preferred: these
/// tests are about that engine, so a file it stops accepting must fail here
/// rather than quietly retarget the whole file at the mirror.
fn open_bounded(path: &std::path::Path) -> Result<File, hdf5_pure::Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
}

use tempfile::tempdir;
use test_util_hdf5::dataset::Unlimited;

/// A dataset `d` (rank-1, unlimited, chunked) plus a `refs` dataset holding one
/// object reference to it.
fn build(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    Unlimited::new("d", &(0..8).collect::<Vec<i32>>(), 4).add_to(&mut b);
    b.create_dataset("refs").with_path_references(&["d"]);
    b.write(path).unwrap();
}

fn deref_dataset(file: &File) -> hdf5_pure::Dataset {
    let mut objs = file.dataset("refs").unwrap().dereference().unwrap();
    match objs.remove(0) {
        Object::Dataset(ds) => *ds,
        other => panic!("expected a dataset, got {other:?}"),
    }
}

/// The capability itself: a path-less handle can append, and the rows land in the
/// live dataset.
#[test]
fn a_dereferenced_handle_can_append() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("byref.h5");
    build(&p);
    {
        let file = File::open_rw(&p).unwrap();
        let mut by_ref = deref_dataset(&file);
        by_ref.append(&[8i32, 9, 10, 11]).unwrap();
        file.close().unwrap();
    }
    assert_eq!(
        File::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        (0..12).collect::<Vec<_>>()
    );
}

/// The same on a bounded file, which is where this capability existed before the
/// engines were merged.
#[test]
fn a_dereferenced_handle_can_append_on_a_bounded_file() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("byref_bounded.h5");
    build(&p);
    {
        let file = open_bounded(&p).unwrap();
        let mut by_ref = deref_dataset(&file);
        by_ref.append(&[8i32, 9, 10, 11]).unwrap();
        file.close().unwrap();
    }
    assert_eq!(
        File::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        (0..12).collect::<Vec<_>>()
    );
}

/// A commit can relocate an object header, and the vacated header still parses —
/// its data-layout message still points at the live chunk index — so an append
/// through a handle that captured the old address would succeed *into the dead
/// header*, grow its dataspace, and report `Ok` while the live dataset stood
/// still. It must be refused instead, with the same [`Error::StaleHandle`] a
/// *read* through that handle now reports (issue #351): one condition, one
/// error, whichever way the handle is used.
#[test]
fn appending_by_reference_after_a_commit_is_refused_not_silently_lost() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("byref_stale.h5");
    build(&p);
    {
        let file = File::open_rw(&p).unwrap();
        let mut by_ref = deref_dataset(&file);
        // Before any commit the address is trustworthy.
        by_ref.append(&[8i32, 9, 10, 11]).unwrap();

        // An attribute edit relocates `d`'s object header.
        file.dataset("d")
            .unwrap()
            .set_attr("tag", AttrValue::I32(1))
            .unwrap();
        file.commit().unwrap();

        let err = by_ref.append(&[12i32, 13, 14, 15]).unwrap_err();
        assert!(
            matches!(err, Error::StaleHandle),
            "unexpected error: {err:?}"
        );

        // Re-fetching by path works, because a path is re-resolved every time.
        file.dataset("d")
            .unwrap()
            .append(&[12i32, 13, 14, 15])
            .unwrap();
        file.close().unwrap();
    }
    assert_eq!(
        File::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        (0..16).collect::<Vec<_>>(),
        "the rows must be in the live dataset, not a vacated header"
    );
}

/// A handle dereferenced *while* edits are already staged is refused too, and by
/// a different rule than the one above: nothing has moved under it, so it is not
/// stale — the session simply cannot check a raw address against edits it has not
/// applied. Two conditions, two errors, each saying which it is.
#[test]
fn appending_by_reference_dereferenced_after_staging_is_refused_by_the_session() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("byref_staged_first.h5");
    build(&p);
    let file = File::open_rw(&p).unwrap();
    file.root().create_group("g").unwrap();
    // Dereferenced now, so its address is the file's current one.
    let mut by_ref = deref_dataset(&file);
    let err = by_ref.append(&[8i32]).unwrap_err();
    assert!(
        matches!(err, Error::AppendInPlaceUnsupported(_)),
        "unexpected error: {err:?}"
    );
}

/// The staged-edit form of the same rule: an address cannot be checked against a
/// pending edit that may move it, so staging one ends this handle just as
/// committing would.
#[test]
fn appending_by_reference_with_staged_edits_pending_is_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("byref_staged.h5");
    build(&p);
    let file = File::open_rw(&p).unwrap();
    let mut by_ref = deref_dataset(&file);
    file.root().create_group("g").unwrap();
    let err = by_ref.append(&[8i32]).unwrap_err();
    assert!(
        matches!(err, Error::StaleHandle),
        "unexpected error: {err:?}"
    );
}
