//! Owned-handle SWMR-writer mode (issue #148, PR B).
//!
//! `File::open_swmr_writer` opens a file for single-writer/multiple-reader
//! appending: no OS lock, the superblock's SWMR-write flag raised while active
//! and cleared on `close`, only immediate `Dataset::append` permitted (over the
//! unfiltered, chunk-aligned SWMR subset), and the staged edit surface refused.

use hdf5_pure::{AttrValue, Error, File};
use tempfile::tempdir;
use test_util::superblock;
use test_util_hdf5::dataset::{Filter, Unlimited};

const SWMR_WRITE_FLAGS: u32 = 0x05;

#[test]
fn swmr_append_reads_back_and_flag_lifecycle() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("s.h5");
    Unlimited::new("d", &(0..4).collect::<Vec<i32>>(), 4).pure_create(&path);

    let file = File::open_swmr_writer(&path).unwrap();
    // The SWMR-write flag is raised on open.
    assert_eq!(superblock::consistency_flags(&path), SWMR_WRITE_FLAGS);
    {
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[4i32, 5, 6, 7]).unwrap(); // one whole chunk
        assert_eq!(ds.read_i32().unwrap(), (0..8).collect::<Vec<_>>());
    }
    // The append preserves the SWMR-write flag; it is cleared only on close.
    assert_eq!(superblock::consistency_flags(&path), SWMR_WRITE_FLAGS);
    file.close().unwrap();
    // A clean close clears the flag.
    assert_eq!(superblock::consistency_flags(&path), 0);
    // The append persisted.
    let ro = File::open(&path).unwrap();
    assert_eq!(
        ro.dataset("d").unwrap().read_i32().unwrap(),
        (0..8).collect::<Vec<_>>()
    );
}

#[test]
fn swmr_refuses_the_staged_surface() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("s.h5");
    Unlimited::new("d", &(0..4).collect::<Vec<i32>>(), 4).pure_create(&path);

    let file = File::open_swmr_writer(&path).unwrap();
    let mut ds = file.dataset("d").unwrap();
    assert!(matches!(
        ds.write(&[9i32, 9, 9, 9]),
        Err(Error::SwmrStagedUnsupported)
    ));
    assert!(matches!(
        ds.set_attr("x", AttrValue::I64(1)),
        Err(Error::SwmrStagedUnsupported)
    ));
    assert!(matches!(
        ds.remove_attr("x"),
        Err(Error::SwmrStagedUnsupported)
    ));
    assert!(matches!(
        ds.append_staged(|b| {
            b.append_i32(&[1, 2, 3, 4]);
        }),
        Err(Error::SwmrStagedUnsupported)
    ));

    let root = file.root();
    assert!(matches!(
        root.create_group("g"),
        Err(Error::SwmrStagedUnsupported)
    ));
    assert!(matches!(
        root.create_dataset("d2", |b| {
            b.with_i32_data(&[1]).with_shape(&[1]);
        }),
        Err(Error::SwmrStagedUnsupported)
    ));
    assert!(matches!(
        root.delete("d"),
        Err(Error::SwmrStagedUnsupported)
    ));
    assert!(matches!(
        root.set_attr("a", AttrValue::I64(1)),
        Err(Error::SwmrStagedUnsupported)
    ));
    assert!(matches!(
        file.copy("d", "d2"),
        Err(Error::SwmrStagedUnsupported)
    ));
    assert!(matches!(file.commit(), Err(Error::SwmrStagedUnsupported)));

    // Immediate append remains allowed.
    ds.append(&[4i32, 5, 6, 7]).unwrap();
}

#[test]
fn swmr_refuses_filtered_and_unaligned_appends() {
    let dir = tempdir().unwrap();

    // Unaligned: length 4 (chunk 4, aligned), append 3 -> not a whole chunk.
    let upath = dir.path().join("u.h5");
    Unlimited::new("d", &(0..4).collect::<Vec<i32>>(), 4).pure_create(&upath);
    {
        let file = File::open_swmr_writer(&upath).unwrap();
        let mut ds = file.dataset("d").unwrap();
        assert!(matches!(
            ds.append(&[4i32, 5, 6]),
            Err(Error::SwmrAppendUnsupported(_))
        ));
    }

    // Filtered: opening is fine (the filter is per dataset), the append is refused.
    let fpath = dir.path().join("f.h5");
    Unlimited::new("d", &(0..4).collect::<Vec<i32>>(), 4)
        .filters(&[Filter::Deflate(6)])
        .pure_create(&fpath);
    let file = File::open_swmr_writer(&fpath).unwrap();
    let mut ds = file.dataset("d").unwrap();
    assert!(matches!(
        ds.append(&[4i32, 5, 6, 7]),
        Err(Error::SwmrAppendUnsupported(_))
    ));
}

#[test]
fn swmr_post_close_append_is_sealed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c.h5");
    Unlimited::new("d", &(0..4).collect::<Vec<i32>>(), 4).pure_create(&path);

    let file = File::open_swmr_writer(&path).unwrap();
    let mut ds = file.dataset("d").unwrap();
    file.close().unwrap();
    assert!(matches!(
        ds.append(&[4i32, 5, 6, 7]),
        Err(Error::FileClosed)
    ));
}

/// A writer that exits without a clean close (simulated by leaking the handle so
/// `Drop` never runs) leaves the flag set; `clear_swmr_flag` recovers it. The
/// leak + exclusive-relock is most predictable on Unix advisory locks.
#[test]
#[cfg(unix)]
fn clear_swmr_flag_recovers_a_stale_flag() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("r.h5");
    Unlimited::new("d", &(0..4).collect::<Vec<i32>>(), 4).pure_create(&path);

    let file = File::open_swmr_writer(&path).unwrap();
    // Drop never runs, so the flag is left set.
    #[expect(clippy::mem_forget, reason = "the test models a writer that crashed")]
    std::mem::forget(file);
    assert_eq!(superblock::consistency_flags(&path), SWMR_WRITE_FLAGS);

    File::clear_swmr_flag(&path).unwrap();
    assert_eq!(superblock::consistency_flags(&path), 0);
}

/// The SWMR writer takes no OS lock, so an exclusive-locking open gets past the
/// lock while it is active — and is then turned away by the superblock's
/// SWMR-write flag instead (issue #245). Which of the two errors comes back is
/// what distinguishes them: `open_rw` takes the lock *before* it reads the
/// superblock, so a writer holding a lock would report `FileLocked` and never
/// reach the flag. Advisory-lock behavior is predictable on Unix; on Windows
/// OS-level file sharing complicates the check.
#[test]
#[cfg(unix)]
fn swmr_holds_no_os_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("l.h5");
    Unlimited::new("d", &(0..4).collect::<Vec<i32>>(), 4).pure_create(&path);

    let writer = File::open_swmr_writer(&path).unwrap();
    let err = File::open_rw(&path).expect_err("a live SWMR writer holds the file");
    assert!(
        matches!(err, Error::FileMarkedInUse(_)),
        "a SWMR writer must not hold an exclusive OS lock, so the refusal must come \
         from the superblock flag rather than the lock; got {err:?}"
    );
    drop(writer);
}
