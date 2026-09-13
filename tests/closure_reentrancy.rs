//! Calling back into the same `File` from inside a builder closure must not
//! deadlock (issue #200).
//!
//! The owned-handle write API takes `&self` plus interior mutability, so the
//! session lives behind a `Mutex`. Holding that mutex across the user's closure
//! made any read from inside it block forever on a lock the same thread already
//! held — no error, no panic, no timeout. The deprecated `EditSession` took
//! `&mut self`, so the same mistake was a borrow-check error; the owned API
//! turned a compiler diagnostic into a lockup.
//!
//! Every test here runs the closure on a worker thread and fails on a timeout,
//! so a regression reports a failure instead of hanging the suite forever.

use hdf5_pure::{AttrValue, File, FileBuilder, MaxExtent};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Run `body` on a worker thread, failing if it has not finished in time.
///
/// A plain call would hang the whole test binary on regression, which reads as
/// a stuck CI job rather than a failing test.
fn without_deadlocking(what: &str, body: impl FnOnce() + Send + 'static) {
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        body();
        let _ = tx.send(());
    });
    match rx.recv_timeout(Duration::from_secs(20)) {
        Ok(()) => handle.join().expect("worker panicked"),
        // A panic in `body` drops the sender, which disconnects rather than
        // times out. Re-raise it so an assertion failure reads as itself and
        // only a genuine hang is reported as a deadlock.
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            std::panic::resume_unwind(handle.join().unwrap_err())
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("{what} deadlocked: the closure never returned")
        }
    }
}

fn seed(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("existing").with_i32_data(&[1, 2, 3, 4]);
    let grp = b.create_group("grp").finish();
    b.add_group(grp);
    b.write(path).unwrap();
}

/// The reproduction from the issue: reading the file from inside
/// `create_dataset`'s closure.
#[test]
fn create_dataset_closure_may_read_the_same_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reentrant_create.h5");
    seed(&path);
    let p = path.clone();

    without_deadlocking("create_dataset", move || {
        let file = File::open_rw(&p).unwrap();
        file.root()
            .create_dataset("derived", |b| {
                // Every one of these reaches the mirror, which is exactly what
                // used to block on the lock the closure was running under.
                let root = file.root();
                let existing = file.dataset("existing").unwrap();
                let values = existing.read_i32().unwrap();
                assert_eq!(values, vec![1, 2, 3, 4]);
                assert!(root.datasets().unwrap().contains(&"existing".to_string()));
                // The staged dataset's contents genuinely depend on the read —
                // the case the issue calls out as natural to write.
                let doubled: Vec<i32> = values.iter().map(|v| v * 2).collect();
                b.with_i32_data(&doubled);
            })
            .unwrap();
        file.commit().unwrap();
    });

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("derived").unwrap().read_i32().unwrap(),
        vec![2, 4, 6, 8]
    );
}

/// The same for the group closure, including a nested one: the nested
/// `StagedGroup` must record into the same buffer rather than re-enter the lock.
#[test]
fn create_group_with_closure_may_read_the_same_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reentrant_group.h5");
    seed(&path);
    let p = path.clone();

    without_deadlocking("create_group_with", move || {
        let file = File::open_rw(&p).unwrap();
        file.root()
            .create_group_with("run2", |g| {
                let n = file.dataset("existing").unwrap().read_i32().unwrap().len();
                g.set_attr("source_len", AttrValue::I64(n as i64));
                g.create_group_with("nested", |inner| {
                    assert!(file.group("grp").is_ok());
                    inner.set_attr("depth", AttrValue::I64(2));
                    inner.create_dataset("leaf", |b| {
                        b.with_i32_data(&[9, 9]);
                    });
                });
            })
            .unwrap();
        file.commit().unwrap();
    });

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.group("run2").unwrap().attrs().unwrap().get("source_len"),
        Some(&AttrValue::I64(4))
    );
    assert_eq!(
        f.group("run2/nested")
            .unwrap()
            .attrs()
            .unwrap()
            .get("depth"),
        Some(&AttrValue::I64(2))
    );
    assert_eq!(
        f.dataset("run2/nested/leaf").unwrap().read_i32().unwrap(),
        vec![9, 9]
    );
}

/// `Dataset::write_staged` and `append_staged` take closures over the same
/// session, so they carried the same hazard.
#[test]
fn dataset_closures_may_read_the_same_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reentrant_dataset.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("fixed").with_i32_data(&[1, 2, 3, 4]);
        b.create_dataset("growable")
            .with_i32_data(&[10, 20])
            .with_chunks(&[2])
            .with_maxshape(&[MaxExtent::Unlimited]);
        b.write(&path).unwrap();
    }
    let p = path.clone();

    without_deadlocking("write_staged / append_staged", move || {
        let file = File::open_rw(&p).unwrap();

        let mut fixed = file.dataset("fixed").unwrap();
        fixed
            .write_staged(|b| {
                let seen = file.dataset("growable").unwrap().read_i32().unwrap();
                assert_eq!(seen, vec![10, 20]);
                b.with_i32_data(&[5, 6, 7, 8]);
            })
            .unwrap();

        let mut growable = file.dataset("growable").unwrap();
        growable
            .append_staged(|b| {
                let seen = file.dataset("fixed").unwrap().read_i32().unwrap();
                // The staged overwrite above is not committed yet, so the read
                // sees the on-disk values — staged edits stay invisible until
                // `commit`, exactly as they do outside a closure.
                assert_eq!(seen, vec![1, 2, 3, 4]);
                b.append(&[30i32, 40]);
            })
            .unwrap();

        file.commit().unwrap();
    });

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("fixed").unwrap().read_i32().unwrap(),
        vec![5, 6, 7, 8]
    );
    assert_eq!(
        f.dataset("growable").unwrap().read_i32().unwrap(),
        vec![10, 20, 30, 40]
    );
}

/// A read-only file must still be refused *before* the closure runs, so the
/// error a caller sees does not depend on whether the fix reordered the work.
#[test]
fn a_read_only_file_is_refused_without_running_the_closure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("read_only.h5");
    seed(&path);

    let file = File::open(&path).unwrap();
    let mut ran = false;
    let err = file
        .root()
        .create_dataset("nope", |b| {
            ran = true;
            b.with_i32_data(&[1]);
        })
        .unwrap_err();
    assert!(matches!(err, hdf5_pure::Error::ReadOnly), "got {err:?}");
    assert!(!ran, "the closure must not run on a read-only file");
}

/// The same property for the other reason a staged edit is refused: a `Dataset`
/// reached by object reference has no resolvable path, so it cannot be edited.
/// Every reason to refuse must be reported before the closure runs, not after —
/// otherwise the caller's closure runs and its work is silently discarded.
#[test]
fn a_pathless_dataset_handle_is_refused_without_running_the_closure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("refs.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("target").with_i32_data(&[1, 2, 3]);
        b.create_dataset("refs").with_path_references(&["target"]);
        b.write(&path).unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let refs = file.dataset("refs").unwrap();
    let mut resolved = refs.dereference().unwrap();
    let hdf5_pure::Object::Dataset(mut reached) = resolved.remove(0) else {
        panic!("expected the reference to resolve to a dataset");
    };

    let mut ran = false;
    let err = reached
        .write_staged(|b| {
            ran = true;
            b.with_i32_data(&[9, 9, 9]);
        })
        .unwrap_err();
    assert!(matches!(err, hdf5_pure::Error::ReadOnly), "got {err:?}");
    assert!(
        !ran,
        "the closure must not run for a handle with no resolvable path"
    );
}
