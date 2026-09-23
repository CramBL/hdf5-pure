//! OS advisory file-locking behavior for the in-place editor (issue #73). Uses
//! only hdf5-pure (no reference C library), so it runs on every target.
//!
//! Only `File::open_rw` takes a lock (exclusive); `File::open_swmr_writer` and the readers take
//! none. Assertions here are written to hold on both Unix (advisory `flock`) and
//! Windows (mandatory `LockFileEx`): in particular we never read a file while an
//! editor still holds its lock, because that read is permitted on Unix but
//! blocked by the OS on Windows.

use hdf5_pure::{Error, File, FileAccessProperties, FileBuilder, FileLocking};
use tempfile::tempdir;
use test_util_hdf5::dataset::Unlimited;

/// A plain, in-place-editable starter file.
fn write_starter(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.write(path).unwrap();
}

/// An unlimited Extensible-Array dataset the SWMR append writer can open.
fn write_appendable(path: &std::path::Path) {
    Unlimited::new("d", &[0i32, 1, 2, 3], 1).pure_create(path);
}

#[test]
fn editor_lock_blocks_a_second_editor() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edit.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap(); // holds the exclusive lock

    // A second editor cannot open the file while the first is alive.
    assert!(
        matches!(File::open_rw(&path), Err(Error::FileLocked(_))),
        "second File::open_rw should be FileLocked"
    );

    drop(session); // releases the lock

    // Once the editor is gone, opening (for edit or read) succeeds again.
    File::open_rw(&path).expect("editor should open after the first is dropped");
    File::open(&path).expect("read should succeed after the editor is dropped");
}

#[test]
fn disabled_locking_takes_no_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edit.h5");
    write_starter(&path);

    // With locking disabled the editor takes no lock at all, so others can still
    // open the file (true on every platform, since nothing is locked).
    let _editor = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    File::open(&path).expect("read should succeed: the Disabled editor took no lock");
    File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .expect("a second Disabled editor should open: neither took a lock");
}

#[test]
fn swmr_writer_does_not_lock_so_a_reader_can_follow_it() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("swmr.h5");
    write_appendable(&path);

    let writer = File::open_swmr_writer(&path).unwrap(); // takes no lock
    writer
        .dataset("d")
        .unwrap()
        .append(&(4..20).collect::<Vec<_>>())
        .unwrap();

    // The writer is still alive: a SWMR reader opens and reads concurrently,
    // which only works because neither side holds an OS lock (on Windows a lock
    // would block the read outright).
    let reader = File::open_swmr(&path).expect("open_swmr should coexist with a live SWMR writer");
    assert_eq!(
        reader.dataset("d").unwrap().read_i32().unwrap(),
        (0..20).collect::<Vec<_>>()
    );

    drop(writer);
}
