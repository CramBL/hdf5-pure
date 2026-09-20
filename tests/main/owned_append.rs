//! Write-through append on an owned handle (issue #148, phase 2).
//!
//! A file opened with `File::open_rw` hands out `Dataset` handles that both read
//! and append in place: `Dataset::append` grows the dataset, and reads through
//! the same handle observe the new length and data. A read-only file refuses the
//! append.

use hdf5_pure::{Error, File, FileBuilder, MaxExtent};
use tempfile::tempdir;

/// Create a rank-1, unlimited i32 dataset with chunk length `chunk`, seeded 0..n.
fn create_i32(path: &std::path::Path, n: i32, chunk: u64) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[chunk]);
    b.write(path).unwrap();
}

#[test]
fn append_and_read_through_one_handle() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 5, 4); // [0,1,2,3,4]
    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![5]);
        ds.append(&[5i32, 6, 7]).unwrap();
        // Read through the SAME handle: it sees the new length and data.
        assert_eq!(ds.shape().unwrap(), vec![8]);
        assert_eq!(ds.read_i32().unwrap(), (0..8).collect::<Vec<_>>());
    } // drop the writer -> releases the exclusive lock before we reopen
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..8).collect::<Vec<_>>()
    );
}

#[test]
fn many_appends_across_calls() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 0, 4);
    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        for v in 0..20i32 {
            ds.append(&[v]).unwrap();
        }
        assert_eq!(ds.read_i32().unwrap(), (0..20).collect::<Vec<_>>());
    }
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..20).collect::<Vec<_>>()
    );
}

#[test]
fn append_to_readonly_file_is_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 3, 4);
    let bytes = std::fs::read(&path).unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let mut ds = file.dataset("d").unwrap();
    assert!(matches!(ds.append(&[9i32]), Err(Error::ReadOnly)));
}

#[test]
fn open_rw_reads_like_open() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 6, 3);
    let file = File::open_rw(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..6).collect::<Vec<_>>()
    );
    assert_eq!(file.dataset("d").unwrap().shape().unwrap(), vec![6]);
}

/// Regression: interleaved read -> append -> read through ONE handle must
/// observe every append. The handle's chunk cache retains the parsed chunk
/// index, and an any-length append relocates the trailing chunk and adds new
/// index entries; before the post-mutation cache clear, the second read served
/// the stale index and returned zeros for the appended elements.
#[test]
fn interleaved_reads_observe_every_append() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("interleaved.h5");
    create_i32(&p, 6, 4);
    let file = File::open_rw(&p).unwrap();
    let mut ds = file.dataset("d").unwrap();
    ds.append(&[6i32, 7, 8]).unwrap();
    assert_eq!(ds.read_i32().unwrap(), (0..9).collect::<Vec<_>>());
    // The read above cached the chunk index; this append repoints the trailing
    // chunk (element 8's chunk moves) and must invalidate that cache.
    ds.append(&[9i32]).unwrap();
    assert_eq!(ds.read_i32().unwrap(), (0..10).collect::<Vec<_>>());
    ds.append(&[10i32, 11, 12, 13]).unwrap();
    assert_eq!(ds.read_i32().unwrap(), (0..14).collect::<Vec<_>>());
}
