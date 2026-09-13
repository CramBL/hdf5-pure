//! `SyncPolicy` — who owns the `fsync` cadence, this crate or the application
//! (issue #263).
//!
//! What the policy governs is invisible from here by construction: a skipped
//! barrier writes exactly the bytes an issued one writes, and this crate's tests
//! simulate a crash by copying the file rather than by losing the page cache, so
//! no test in this suite can observe the difference. That the barriers really are
//! skipped is asserted in the library tests, through an image that counts them.
//!
//! These are the other half — that skipping them costs durability and *nothing
//! else*: the same file, the same content, the same refusals.

use hdf5_pure::{
    AttrValue, Error, File, FileAccessProperties, FileBuilder, FileSpaceStrategy, MaxExtent,
    SyncPolicy,
};
use tempfile::tempdir;

/// A file with one unlimited chunked dataset, so a session can reach it by both
/// an immediate append and a staged commit.
fn fixture(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..8).collect::<Vec<_>>())
        .with_shape(&[8])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    b.write(path).unwrap();
}

/// Drive one read-write session through every write path there is, under
/// `policy`, and return the file it produced.
fn write_everything(path: &std::path::Path, policy: SyncPolicy) -> Vec<u8> {
    fixture(path);
    let file =
        File::open_rw_with_options(path, FileAccessProperties::new().with_sync_policy(policy))
            .unwrap();

    // Immediate: the ordered, barriered in-place append.
    file.dataset("d")
        .unwrap()
        .append(&[8i32, 9, 10, 11])
        .unwrap();
    // Staged: a new group, a new dataset, and an attribute, applied by commits.
    file.root().create_group("g").unwrap();
    file.commit().unwrap();
    file.group("g")
        .unwrap()
        .create_dataset("inner", |b| {
            b.with_f64_data(&[2.5; 4]).with_shape(&[4]);
        })
        .unwrap();
    file.root().set_attr("tag", AttrValue::I64(7)).unwrap();
    file.commit().unwrap();

    // A mid-session checkpoint under either policy; `close` issues the closing
    // barrier itself. Either way it writes nothing.
    file.sync().unwrap();
    file.close().unwrap();
    std::fs::read(path).unwrap()
}

/// The policy changes when bytes are forced to the platter, not which bytes.
///
/// Byte identity is the strongest form of that claim available: a barrier that
/// also, say, settled an allocation would show up here as a difference, where
/// reading the content back would not.
#[test]
fn on_close_writes_the_same_file_as_always() {
    let dir = tempdir().unwrap();
    let always = write_everything(&dir.path().join("always.h5"), SyncPolicy::Always);
    let deferred = write_everything(&dir.path().join("on_close.h5"), SyncPolicy::OnClose);
    assert_eq!(
        always.len(),
        deferred.len(),
        "a file written with no fsync must be the same size as one written with them"
    );
    assert!(
        always == deferred,
        "a file written with no fsync must be byte-identical to one written with them"
    );

    // And it is a working HDF5 file, not merely an identical one.
    let f = File::open(dir.path().join("on_close.h5")).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..12).collect::<Vec<_>>()
    );
    assert_eq!(
        f.dataset("g/inner").unwrap().read_f64().unwrap(),
        vec![2.5; 4]
    );
    assert_eq!(
        f.root().attrs().unwrap().get("tag").unwrap().as_i64(),
        Some(7)
    );
}

/// `sync` is a write-path operation: a read-only file has nothing to make
/// durable and says so, rather than reporting a success it did not perform.
#[test]
fn sync_is_refused_on_a_read_only_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ro.h5");
    fixture(&path);

    let file = File::open(&path).unwrap();
    assert!(
        matches!(file.sync(), Err(Error::ReadOnly)),
        "a read-only file must refuse sync"
    );
}

/// A sealed file refuses `sync` like every other write-path method — and needs
/// no sync, because `close` issued one. This is the pair that makes `OnClose`
/// a contract with no recipe attached: the caller cannot sync a closed file, and
/// does not have to.
#[test]
fn a_closed_file_refuses_sync_and_does_not_need_one() {
    let dir = tempdir().unwrap();

    // A file that persists its free space, so `close` has manager re-homing to
    // do after the last barrier the caller could have asked for.
    let path = dir.path().join("persist.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..8).collect::<Vec<_>>())
        .with_shape(&[8])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    file.dataset("d")
        .unwrap()
        .append(&[8i32, 9, 10, 11])
        .unwrap();
    file.root().create_group("g").unwrap();

    let handle = file.clone();
    file.close().unwrap();
    assert!(
        matches!(handle.sync(), Err(Error::FileClosed)),
        "a closed file must refuse sync; close already issued the barrier"
    );
    assert!(
        matches!(handle.root().create_group("late"), Err(Error::FileClosed)),
        "and must still be sealed against writes"
    );

    drop(handle);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..12).collect::<Vec<_>>()
    );
    assert!(
        f.group("g").is_ok(),
        "close must have applied the staged group"
    );
}

/// The default is the durable one: an application that never mentions the policy
/// keeps every barrier it had before this setting existed.
#[test]
fn the_default_policy_is_always() {
    assert_eq!(
        FileAccessProperties::new().sync_policy(),
        SyncPolicy::Always
    );
    assert_eq!(
        FileAccessProperties::default().sync_policy(),
        SyncPolicy::Always
    );
    assert_eq!(
        FileAccessProperties::new()
            .with_sync_policy(SyncPolicy::OnClose)
            .sync_policy(),
        SyncPolicy::OnClose,
        "and the accessor reports what was asked for"
    );
}
