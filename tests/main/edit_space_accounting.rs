//! Live space accounting on [`File::open_rw`] (issue #150).
//!
//! [`File::space_accounting`] reports the file *as the session currently
//! holds it*: its live logical size and the free space it can reuse. These tests
//! pin the semantics the docs promise — the size tracks immediate in-place
//! appends, reusable free reflects only committed frees (not staged edits nor a
//! fresh non-persisting session's pre-existing holes), a persisting session seeds
//! its reusable free from disk on open, and the total always equals the summed
//! region lengths.

use hdf5_pure::DeflateLevel;
use hdf5_pure::{File, FileBuilder};
use test_util::range;
use test_util_hdf5::storage;
use test_util_hdf5::{
    dataset::{Filter, Unlimited},
    file_space,
};

#[test]
fn fresh_session_reports_file_size_and_no_reusable_free() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("plain.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    b.write(&p).unwrap();

    let on_disk = std::fs::metadata(&p).unwrap().len();
    let s = File::open_rw(&p).unwrap();
    let acct = s.space_accounting().unwrap();

    assert_eq!(
        acct.logical_size, on_disk,
        "logical_size must equal the physical file length"
    );
    assert_eq!(
        acct.reusable_free_bytes, 0,
        "a freshly opened non-persisting session tracks no reusable free space"
    );
    assert!(acct.reusable_free_space.is_empty());
    file_space::assert_space_accounting_consistent(&acct);
}

#[test]
fn logical_size_grows_with_immediate_append() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("append.h5");
    Unlimited::new("d", &(0..8).collect::<Vec<i32>>(), 4).pure_create(&p);

    let s = File::open_rw(&p).unwrap();
    let before = s.space_accounting().unwrap().logical_size;

    // An immediate, durable in-place append: the mirror grows and stays in lockstep
    // with the on-disk file, so logical_size reflects it at once.
    s.dataset("d")
        .unwrap()
        .append(&[8, 9, 10, 11, 12, 13, 14, 15])
        .unwrap();
    let acct = s.space_accounting().unwrap();

    assert!(
        acct.logical_size > before,
        "an in-place append must grow the live logical size ({before} -> {})",
        acct.logical_size
    );
    assert_eq!(
        acct.logical_size,
        std::fs::metadata(&p).unwrap().len(),
        "logical_size must match the on-disk file length after a durable append"
    );
    assert_eq!(
        acct.reusable_free_space,
        [(before + 32, 2016)],
        "an immediate append retains its unused raw aggregate reservation"
    );
    file_space::assert_space_accounting_consistent(&acct);
}

#[test]
fn staged_delete_is_not_counted_until_commit() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("staged.h5");
    file_space::write_interior_hole_source(&p, false);

    let s = File::open_rw(&p).unwrap();
    assert!(!s.has_staged_edits());
    assert_eq!(s.space_accounting().unwrap().reusable_free_bytes, 0);

    // Staging the delete must not change the accounting: bytes are freed at commit.
    s.root().delete("big").unwrap();
    assert!(s.has_staged_edits());
    assert_eq!(
        s.space_accounting().unwrap().reusable_free_bytes,
        0,
        "a staged (uncommitted) delete frees nothing yet"
    );

    // Committing frees `big`'s interior storage, which becomes reusable.
    s.commit().unwrap();
    assert!(!s.has_staged_edits());
    let acct = s.space_accounting().unwrap();
    assert!(
        acct.reusable_free_bytes >= 1600,
        "committing the delete of an interior dataset makes its ~1600 bytes reusable \
         (got {})",
        acct.reusable_free_bytes
    );
    file_space::assert_space_accounting_consistent(&acct);
}

#[test]
fn reused_free_shrinks_the_reusable_total() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("reuse.h5");
    file_space::write_interior_hole_source(&p, false);

    {
        let s = File::open_rw(&p).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
        let free_before = s.space_accounting().unwrap().reusable_free_bytes;
        assert!(free_before >= 1600);

        // A new dataset that fits the freed hole reuses it rather than growing the
        // file, so the reusable total drops.
        s.root()
            .create_dataset("d", |b| {
                b.with_i32_data(&[9; 300]);
            })
            .unwrap(); // 1200 bytes, fits the hole
        s.commit().unwrap();
        let acct = s.space_accounting().unwrap();
        assert!(
            acct.reusable_free_bytes < free_before,
            "reusing the hole must shrink the reusable total ({free_before} -> {})",
            acct.reusable_free_bytes
        );
        file_space::assert_space_accounting_consistent(&acct);
    }

    // Reopen only after the session (and its exclusive file lock) is dropped —
    // Windows enforces the lock against a concurrent `File::open`.
    let f = File::open(&p).unwrap();
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![9; 300]);
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
}

#[test]
fn fresh_nonpersisting_session_ignores_existing_holes() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("holes.h5");
    file_space::write_interior_hole_source(&p, false);

    // Session 1 leaves an interior hole on disk (untracked, since not persisting).
    {
        let s = File::open_rw(&p).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
    }

    // Session 2, freshly opened, does not scan for or track those holes.
    let s = File::open_rw(&p).unwrap();
    let acct = s.space_accounting().unwrap();
    assert_eq!(
        acct.reusable_free_bytes, 0,
        "a non-persisting reopen ignores holes left by a prior session"
    );
    assert!(acct.reusable_free_space.is_empty());
    assert_eq!(acct.logical_size, std::fs::metadata(&p).unwrap().len());
    file_space::assert_space_accounting_consistent(&acct);
}

#[test]
fn persisting_session_seeds_reusable_free_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("persist.h5");
    file_space::write_interior_hole_source(&p, true);

    // Session 1: delete `big`; with persistence on, its storage is recorded on disk.
    {
        let s = File::open_rw(&p).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
    }

    // What a plain reader recovers from the on-disk managers is the yardstick.
    let persisted: u64 = File::open(&p)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    assert!(
        persisted >= 1600,
        "the delete must persist as free space on disk"
    );

    // Session 2: opening seeds the free list from those managers, so the accounting
    // reports the reusable space immediately, before any edit in this session.
    let s = File::open_rw(&p).unwrap();
    let acct = s.space_accounting().unwrap();
    assert_eq!(
        acct.reusable_free_bytes, persisted,
        "a persisting session seeds its reusable free from disk to match the reader"
    );
    assert!(!acct.reusable_free_space.is_empty());
    file_space::assert_space_accounting_consistent(&acct);
}

#[test]
fn a_relocating_overwrite_reclaims_the_old_header_with_its_storage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("relocate.h5");

    // Zeros compress to almost nothing, so the replacement below cannot fit back
    // into the chunk slots and the dataset is rebuilt elsewhere.
    Unlimited::new("d", &[0i32; 64], 16)
        .filters(&[Filter::Deflate(DeflateLevel::Six)])
        .pure_create(&path);
    let original_end = std::fs::metadata(&path).unwrap().len();

    let bytes = std::fs::read(&path).unwrap();
    let reader = File::open(&path).unwrap();
    let header = storage::root_object_address(&reader, &bytes, "d");
    let mut original = storage::header_extents(&reader, &bytes, header);
    original.extend(
        file_space::raw_extents(&reader, "d")
            .iter()
            .map(|extent| (extent.start, extent.end - extent.start)),
    );
    reader.close().unwrap();
    let session = File::open_rw(&path).unwrap();
    let incompressible: Vec<i32> = (0..64)
        .map(|i: i32| i.wrapping_mul(0x9E37_79B1u32 as i32))
        .collect();
    session
        .dataset("d")
        .unwrap()
        .write_staged(|b| {
            b.with_i32_data(&incompressible);
        })
        .unwrap();
    session.commit().unwrap();

    let acct = session.space_accounting().unwrap();
    file_space::assert_space_accounting_consistent(&acct);
    for (start, len) in original {
        assert_eq!(
            range::covered_len(&acct.reusable_free_space, &(start..start + len)),
            len,
            "old allocation {start}..{} is not fully reclaimed: {:?}",
            start + len,
            acct.reusable_free_space,
        );
    }
    assert_eq!(
        acct.reusable_free_space
            .iter()
            .filter(|&&(start, _)| start < original_end)
            .count(),
        2,
    );

    drop(session);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        incompressible
    );
}
