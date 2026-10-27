//! Tests session-reusable space statistics and sections.
//!
//! Checks immediate appends, staged and committed frees, persistence across sessions, and agreement
//! between scalar totals and section lengths.

use hdf5_pure::DeflateLevel;
use hdf5_pure::FreeSectionCoordinates;
use hdf5_pure::{File, FileBuilder};
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
    let stats = s.reusable_space_stats().unwrap();
    let sections = s.reusable_free_sections().unwrap();

    assert_eq!(
        stats.logical_size(),
        on_disk,
        "logical_size must equal the physical file length"
    );
    assert_eq!(
        stats.reusable_free_bytes(),
        0,
        "a freshly opened non-persisting session tracks no reusable free space"
    );
    assert_eq!(sections.section_count(), 0);
    test_util_hdf5::assert_free_sections_consistent!(
        &stats,
        &sections,
        FreeSectionCoordinates::Absolute
    );
}

#[test]
fn logical_size_grows_with_immediate_append() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("append.h5");
    Unlimited::new("d", &(0..8).collect::<Vec<i32>>(), 4).pure_create(&p);

    let s = File::open_rw(&p).unwrap();
    let before = s.reusable_space_stats().unwrap().logical_size();

    // An immediate, durable in-place append: the mirror grows and stays in lockstep
    // with the on-disk file, so logical_size reflects it at once.
    s.dataset("d")
        .unwrap()
        .append(&[8, 9, 10, 11, 12, 13, 14, 15])
        .unwrap();
    let stats = s.reusable_space_stats().unwrap();
    let sections = s.reusable_free_sections().unwrap();

    assert!(
        stats.logical_size() > before,
        "an in-place append must grow the live logical size ({before} -> {})",
        stats.logical_size()
    );
    assert_eq!(
        stats.logical_size(),
        std::fs::metadata(&p).unwrap().len(),
        "logical_size must match the on-disk file length after a durable append"
    );
    test_util_hdf5::assert_free_sections!(
        &sections,
        &[(before + 32, 2016)],
        FreeSectionCoordinates::Absolute
    );
    test_util_hdf5::assert_free_sections_consistent!(
        &stats,
        &sections,
        FreeSectionCoordinates::Absolute
    );
}

#[test]
fn staged_delete_is_not_counted_until_commit() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("staged.h5");
    file_space::write_interior_hole_source(&p, false);

    let s = File::open_rw(&p).unwrap();
    assert!(!s.has_staged_edits());
    assert_eq!(s.reusable_space_stats().unwrap().reusable_free_bytes(), 0);

    // Staging the delete must not change the accounting: bytes are freed at commit.
    s.root().delete("big").unwrap();
    assert!(s.has_staged_edits());
    assert_eq!(
        s.reusable_space_stats().unwrap().reusable_free_bytes(),
        0,
        "a staged (uncommitted) delete frees nothing yet"
    );

    // Committing frees `big`'s interior storage, which becomes reusable.
    s.commit().unwrap();
    assert!(!s.has_staged_edits());
    let stats = s.reusable_space_stats().unwrap();
    let sections = s.reusable_free_sections().unwrap();
    assert!(
        stats.reusable_free_bytes() >= 1600,
        "committing the delete of an interior dataset makes its ~1600 bytes reusable \
         (got {})",
        stats.reusable_free_bytes()
    );
    test_util_hdf5::assert_free_sections_consistent!(
        &stats,
        &sections,
        FreeSectionCoordinates::Absolute
    );
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
        let free_before = s.reusable_space_stats().unwrap().reusable_free_bytes();
        assert!(free_before >= 1600);

        // A new dataset that fits the freed hole reuses it rather than growing the
        // file, so the reusable total drops.
        s.root()
            .create_dataset("d", |b| {
                b.with_i32_data(&[9; 300]);
            })
            .unwrap(); // 1200 bytes, fits the hole
        s.commit().unwrap();
        let stats = s.reusable_space_stats().unwrap();
        let sections = s.reusable_free_sections().unwrap();
        assert!(
            stats.reusable_free_bytes() < free_before,
            "reusing the hole must shrink the reusable total ({free_before} -> {})",
            stats.reusable_free_bytes()
        );
        test_util_hdf5::assert_free_sections_consistent!(
            &stats,
            &sections,
            FreeSectionCoordinates::Absolute
        );
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
    let stats = s.reusable_space_stats().unwrap();
    let sections = s.reusable_free_sections().unwrap();
    assert_eq!(
        stats.reusable_free_bytes(),
        0,
        "a non-persisting reopen ignores holes left by a prior session"
    );
    assert_eq!(sections.section_count(), 0);
    assert_eq!(stats.logical_size(), std::fs::metadata(&p).unwrap().len());
    test_util_hdf5::assert_free_sections_consistent!(
        &stats,
        &sections,
        FreeSectionCoordinates::Absolute
    );
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
    let stats = s.reusable_space_stats().unwrap();
    let sections = s.reusable_free_sections().unwrap();
    assert_eq!(
        stats.reusable_free_bytes(),
        persisted,
        "a persisting session seeds its reusable free from disk to match the reader"
    );
    assert!(sections.section_count() > 0);
    test_util_hdf5::assert_free_sections_consistent!(
        &stats,
        &sections,
        FreeSectionCoordinates::Absolute
    );
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
    let mut original = storage::header_extents(reader.superblock(), &bytes, header);
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

    let stats = session.reusable_space_stats().unwrap();
    let sections = session.reusable_free_sections().unwrap();
    test_util_hdf5::assert_free_sections_consistent!(
        &stats,
        &sections,
        FreeSectionCoordinates::Absolute
    );
    for (start, len) in original {
        assert_eq!(
            test_util_hdf5::covered_free_bytes!(
                &sections,
                &(start..start + len),
                FreeSectionCoordinates::Absolute
            ),
            len,
            "old allocation {start}..{} is not fully reclaimed: {:?}",
            start + len,
            sections,
        );
    }
    assert_eq!(
        sections
            .sections()
            .iter()
            .filter(|section| section.address() < original_end)
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
