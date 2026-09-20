//! Owned, cacheable object handles (issue #148).
//!
//! `Dataset`, `Group`, and `Object` no longer borrow `File`: they share
//! ownership of the open file, so a handle can be returned from a function that
//! owns its `File`, stored in a struct with no lifetime, cached, cloned, and
//! moved across threads.

use hdf5_pure::{Dataset, Error, File, FileBuilder, Group, Object};

fn sample_file() -> Vec<u8> {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
        .with_shape(&[4]);
    builder.finish().unwrap()
}

/// Owns a `File` and returns a `Dataset` derived from it. This does not compile
/// under a borrow-based `Dataset<'f>` — the returned handle would borrow a local
/// — and is the core ergonomic the issue asks for.
fn open_owned(bytes: Vec<u8>) -> Result<Dataset, Error> {
    let file = File::from_bytes(bytes)?;
    file.dataset("data")
}

#[test]
fn dataset_outlives_the_file_binding() {
    let ds = open_owned(sample_file()).unwrap();
    // `file` inside `open_owned` is long gone; the handle kept the file alive.
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// A struct holds a handle with no lifetime parameter.
struct Cache {
    ds: Dataset,
}

#[test]
fn handle_stored_in_a_struct() {
    let file = File::from_bytes(sample_file()).unwrap();
    let cache = Cache {
        ds: file.dataset("data").unwrap(),
    };
    drop(file); // the cached handle keeps the file open
    assert_eq!(cache.ds.shape().unwrap(), vec![4]);
}

#[test]
fn handles_are_send_sync_static() {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<File>();
    assert_send_sync_static::<Dataset>();
    assert_send_sync_static::<Group>();
    assert_send_sync_static::<Object>();
}

#[test]
fn file_is_cloneable_and_both_clones_read() {
    let file = File::from_bytes(sample_file()).unwrap();
    let file2 = file.clone();
    let expect = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(file.dataset("data").unwrap().read_f64().unwrap(), expect);
    assert_eq!(file2.dataset("data").unwrap().read_f64().unwrap(), expect);
}

#[test]
fn two_handles_to_one_dataset_read_consistently() {
    let file = File::from_bytes(sample_file()).unwrap();
    let a = file.dataset("data").unwrap();
    let b = file.dataset("data").unwrap();
    assert_eq!(a.read_f64().unwrap(), b.read_f64().unwrap());
}

#[test]
fn refresh_reports_outstanding_handles() {
    let mut file = File::from_bytes(sample_file()).unwrap();
    let ds = file.dataset("data").unwrap();
    // A live handle shares ownership, so a mutating refresh is refused with a
    // clear error rather than mutating a view others hold.
    assert!(matches!(file.refresh(), Err(Error::HandlesOutstanding)));
    drop(ds);
    // With the handle dropped, refresh has exclusive access and now fails only
    // because this file was not opened for SWMR.
    assert!(matches!(file.refresh(), Err(Error::SwmrUnsupported)));
}

#[test]
fn handle_moves_across_threads() {
    let file = File::from_bytes(sample_file()).unwrap();
    let ds = file.dataset("data").unwrap();
    let sum = std::thread::spawn(move || ds.read_f64().unwrap().iter().sum::<f64>())
        .join()
        .unwrap();
    assert_eq!(sum, 10.0);
}
