//! Owned-handle SWMR-writer mode (issue #148, PR B).
//!
//! `File::open_swmr_writer` opens a file for single-writer/multiple-reader
//! appending: no OS lock, the superblock's SWMR-write flag raised while active
//! and cleared on `close`, only immediate `Dataset::append` permitted (over the
//! unfiltered, chunk-aligned SWMR subset), and the staged edit surface refused.

use hdf5_pure::DeflateLevel;
use hdf5_pure::{AttrValue, Error, File};

use test_util::superblock;
use test_util_hdf5::dataset::{Filter, Unlimited};

const SWMR_WRITE_FLAGS: u32 = 0x05;

#[test]
fn swmr_append_reads_back_and_flag_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
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
    let dir = tempfile::tempdir().unwrap();
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

#[rstest::rstest]
#[case::unaligned(&[], &[4, 5, 6], "SWMR append must be chunk-aligned: the current length and the appended length must both be whole multiples of the chunk length")]
#[case::filtered(&[Filter::Deflate(DeflateLevel::Six)], &[4, 5, 6, 7], "filtered datasets are not supported for SWMR append")]
fn swmr_refuses_filtered_and_unaligned_appends(
    #[case] filters: &[Filter],
    #[case] values: &[i32],
    #[case] expected: &str,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.h5");
    Unlimited::new("d", &(0..4).collect::<Vec<i32>>(), 4)
        .filters(filters)
        .pure_create(&path);
    let file = File::open_swmr_writer(&path).unwrap();
    let mut ds = file.dataset("d").unwrap();
    let error = ds.append(values).unwrap_err();
    let Error::SwmrAppendUnsupported(actual) = error else {
        panic!("{error:?}");
    };
    assert_eq!(actual, expected);
}

#[test]
fn swmr_post_close_append_is_sealed() {
    let dir = tempfile::tempdir().unwrap();
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

#[test]
#[cfg(unix)]
fn clear_status_flags_recovers_a_stale_flag() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r.h5");
    Unlimited::new("d", &(0..4).collect::<Vec<i32>>(), 4).pure_create(&path);

    let file = File::open_swmr_writer(&path).unwrap();
    // Drop never runs, so the flag is left set.
    #[expect(clippy::mem_forget, reason = "the test models a writer that crashed")]
    std::mem::forget(file);
    assert_eq!(superblock::consistency_flags(&path), SWMR_WRITE_FLAGS);

    File::clear_status_flags(&path).unwrap();
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
    let dir = tempfile::tempdir().unwrap();
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
