// Reads back with the reference HDF5 C library (`hdf5-metno`), gated to
// 64-bit little-endian targets; skip elsewhere so `cross test` on `i686` and
// `s390x` stays pure-Rust.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! SWMR append-writer tests: hdf5-pure appends in place to an unlimited
//! Extensible-Array dataset, and the result is read back by hdf5-pure and by the
//! reference C library. Appends cross the inline -> direct-block -> super-block
//! boundaries so the in-place index growth is exercised.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{Error, File, FileBuilder};
use tempfile::tempdir;

fn pure_create(path: &std::path::Path, n: usize) {
    let data: Vec<i32> = (0..n as i32).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[1]);
    b.write(path).unwrap();
}

fn c_create(path: &std::path::Path, n: usize) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .chunk((1,))
        .shape((Extent::resizable(n),))
        .create("d")
        .unwrap();
    let data: Vec<i32> = (0..n as i32).collect();
    ds.write(&data).unwrap();
    file.close().unwrap();
}

fn read_pure(path: &std::path::Path) -> Vec<i32> {
    let f = File::from_bytes(std::fs::read(path).unwrap()).unwrap();
    f.dataset("d").unwrap().read_i32().unwrap()
}

fn read_c(path: &std::path::Path) -> Vec<i32> {
    let f = hdf5::File::open(path).unwrap();
    f.dataset("d").unwrap().read_raw::<i32>().unwrap()
}

/// Append to an hdf5-pure-created file, crossing every structural boundary, and
/// confirm both hdf5-pure and the C library read the full result.
#[test]
fn append_to_pure_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    pure_create(&path, 10);

    {
        let w = File::open_swmr_writer(&path).unwrap();
        // 10 -> 100 (fills direct data blocks)
        w.dataset("d")
            .unwrap()
            .append(&(10..100).collect::<Vec<_>>())
            .unwrap();
        // 100 -> 300 (crosses into the first super block)
        w.dataset("d")
            .unwrap()
            .append(&(100..300).collect::<Vec<_>>())
            .unwrap();
        // 300 -> 5000 (deeper super-block nesting)
        w.dataset("d")
            .unwrap()
            .append(&(300..5000).collect::<Vec<_>>())
            .unwrap();
    }

    let expected: Vec<i32> = (0..5000).collect();
    assert_eq!(read_pure(&path), expected, "hdf5-pure read mismatch");
    assert_eq!(read_c(&path), expected, "C-library read mismatch");
}

/// Append to a C-library-created file and confirm both readers agree.
#[test]
fn append_to_c_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    c_create(&path, 10);

    {
        let w = File::open_swmr_writer(&path).unwrap();
        w.dataset("d")
            .unwrap()
            .append(&(10..1000).collect::<Vec<_>>())
            .unwrap();
    }

    let expected: Vec<i32> = (0..1000).collect();
    assert_eq!(read_c(&path), expected, "C-library read mismatch");
    assert_eq!(read_pure(&path), expected, "hdf5-pure read mismatch");
}

/// End-to-end SWMR loop within hdf5-pure: a refreshing reader follows the
/// append writer's in-place appends (separate file handles, same file).
#[test]
fn refreshing_reader_follows_pure_appends() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    pure_create(&path, 10);

    let mut reader = File::open_swmr(&path).unwrap();
    assert_eq!(
        reader.dataset("d").unwrap().read_i32().unwrap(),
        (0..10).collect::<Vec<_>>()
    );

    let w = File::open_swmr_writer(&path).unwrap();
    w.dataset("d")
        .unwrap()
        .append(&(10..300).collect::<Vec<_>>())
        .unwrap();

    reader.refresh().unwrap();
    {
        let ds = reader.dataset("d").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![300]);
        assert_eq!(ds.read_i32().unwrap(), (0..300).collect::<Vec<_>>());
    }

    // Another round of appends, then refresh again.
    w.dataset("d")
        .unwrap()
        .append(&(300..900).collect::<Vec<_>>())
        .unwrap();
    reader.refresh().unwrap();
    assert_eq!(
        reader.dataset("d").unwrap().read_i32().unwrap(),
        (0..900).collect::<Vec<_>>()
    );
}

/// Append across the paged-data-block boundary (131060 chunks): start just
/// below it (built by the bulk writer for speed), then append past it so the
/// writer must allocate a paged super block and paged data blocks in place.
#[test]
fn append_crosses_paging_boundary() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let start = 131_000usize;
    let end = 135_000usize;
    {
        let data: Vec<i32> = (0..start as i32).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[start as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[1]);
        b.write(&path).unwrap();
    }
    {
        let w = File::open_swmr_writer(&path).unwrap();
        w.dataset("d")
            .unwrap()
            .append(&(start as i32..end as i32).collect::<Vec<_>>())
            .unwrap();
    }
    let expected: Vec<i32> = (0..end as i32).collect();
    assert_eq!(
        read_pure(&path),
        expected,
        "hdf5-pure read mismatch (paged)"
    );
    assert_eq!(read_c(&path), expected, "C-library read mismatch (paged)");
}

/// A non-latest-format (v0/v1 superblock) file must be rejected with a clear
/// error and left byte-for-byte unchanged. Regression for the bug where
/// `open()` wrote the SWMR flag through `Superblock::serialize` (which always
/// emits the v2/v3 layout), clobbering a v0/v1 superblock before any append.
#[test]
fn rejects_and_preserves_non_latest_format_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("legacy.h5");
    // Force the earliest format so the file gets a v0/v1 superblock.
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::Earliest, LibraryVersion::V18))
            .create(&path)
            .unwrap();
        let ds = file.new_dataset::<i32>().shape((5,)).create("d").unwrap();
        ds.write(&(0..5).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let before = std::fs::read(&path).unwrap();
    let sig = b"\x89HDF\r\n\x1a\n";
    let off = before.windows(8).position(|w| w == sig).unwrap();
    assert!(
        before[off + 8] < 2,
        "test precondition: expected a v0/v1 superblock, got version {}",
        before[off + 8]
    );

    let err = match File::open_swmr_writer(&path) {
        Ok(_) => panic!("expected open() to reject a v0/v1 superblock file"),
        Err(e) => e,
    };
    assert!(
        matches!(err, hdf5_pure::Error::SwmrAppendUnsupported(_)),
        "expected SwmrAppendUnsupported, got {err:?}"
    );

    let after = std::fs::read(&path).unwrap();
    assert_eq!(before, after, "open() must not mutate a rejected file");
    assert_eq!(read_c(&path), (0..5).collect::<Vec<i32>>());
}

/// f64 dataset append, just to exercise a non-4-byte element size.
#[test]
fn append_f64_pure_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("f.h5");
    {
        let data: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[5])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[1]);
        b.write(&path).unwrap();
    }
    {
        let w = File::open_swmr_writer(&path).unwrap();
        let more: Vec<f64> = (5..400).map(|i| i as f64).collect();
        w.dataset("d").unwrap().append(&more).unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    let v = f.dataset("d").unwrap().read_raw::<f64>().unwrap();
    let expected: Vec<f64> = (0..400).map(|i| i as f64).collect();
    assert_eq!(v, expected);
}

/// Every other fixture uses a chunk length of 1, where "element count" and
/// "chunk count" collapse and a chunk-vs-element confusion is invisible. This
/// exercises a multi-element chunk (`chunks = [4]`): aligned appends that cross
/// the inline -> direct -> super-block boundary must read back correctly via
/// pure and C, and an append whose length is not a whole number of chunks must
/// be rejected before any write.
#[test]
fn append_chunk_size_greater_than_one() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("chunk4.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..16).collect::<Vec<_>>())
        .with_shape(&[16])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();

    {
        let w = File::open_swmr_writer(&path).unwrap();
        // Aligned (multiples of the chunk length), crossing inline -> direct -> super.
        w.dataset("d")
            .unwrap()
            .append(&(16..240).collect::<Vec<_>>())
            .unwrap();
        w.dataset("d")
            .unwrap()
            .append(&(240..400).collect::<Vec<_>>())
            .unwrap();

        // Unaligned append (2 elements, 2 % 4 != 0) is rejected before any write.
        let err = match w.dataset("d").unwrap().append(&[400, 401]) {
            Ok(()) => panic!("expected an unaligned append to be rejected"),
            Err(e) => e,
        };
        assert!(
            matches!(err, Error::SwmrAppendUnsupported(_)),
            "expected SwmrAppendUnsupported for an unaligned append, got {err:?}"
        );
    }

    let expected: Vec<i32> = (0..400).collect();
    assert_eq!(
        read_pure(&path),
        expected,
        "hdf5-pure read mismatch (chunk=4)"
    );
    assert_eq!(read_c(&path), expected, "C-library read mismatch (chunk=4)");
}

/// Appending to one dataset must not disturb a sibling. No other test has more
/// than one dataset, so cross-dataset isolation (correct object-header
/// resolution, and EOF appends not clobbering another dataset's blocks) is
/// otherwise unverified. The shared superblock legitimately changes, so the
/// sibling is checked by reading it back, not byte-for-byte.
#[test]
fn append_to_one_of_multiple_datasets_leaves_others_intact() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("multi.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("a")
        .with_i32_data(&(0..10).collect::<Vec<_>>())
        .with_shape(&[10])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[1]);
    b.create_dataset("b")
        .with_i32_data(&(100..110).collect::<Vec<_>>())
        .with_shape(&[10])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[1]);
    b.write(&path).unwrap();

    let read_pure_named = |name: &str| -> Vec<i32> {
        let f = File::from_bytes(std::fs::read(&path).unwrap()).unwrap();
        f.dataset(name).unwrap().read_i32().unwrap()
    };
    let read_c_named = |name: &str| -> Vec<i32> {
        let f = hdf5::File::open(&path).unwrap();
        f.dataset(name).unwrap().read_raw::<i32>().unwrap()
    };

    // Append into "a", crossing into the super blocks; "b" must be untouched.
    {
        let w = File::open_swmr_writer(&path).unwrap();
        w.dataset("a")
            .unwrap()
            .append(&(10..300).collect::<Vec<_>>())
            .unwrap();
    }
    assert_eq!(read_pure_named("a"), (0..300).collect::<Vec<_>>());
    assert_eq!(
        read_pure_named("b"),
        (100..110).collect::<Vec<_>>(),
        "sibling b changed (pure)"
    );
    assert_eq!(
        read_c_named("b"),
        (100..110).collect::<Vec<_>>(),
        "sibling b changed (C)"
    );

    // Now append into "b"; "a" must be untouched.
    {
        let w = File::open_swmr_writer(&path).unwrap();
        w.dataset("b")
            .unwrap()
            .append(&(110..400).collect::<Vec<_>>())
            .unwrap();
    }
    assert_eq!(
        read_pure_named("a"),
        (0..300).collect::<Vec<_>>(),
        "sibling a changed (pure)"
    );
    assert_eq!(read_pure_named("b"), (100..400).collect::<Vec<_>>());
    assert_eq!(
        read_c_named("a"),
        (0..300).collect::<Vec<_>>(),
        "sibling a changed (C)"
    );
    assert_eq!(read_c_named("b"), (100..400).collect::<Vec<_>>());
}

/// Reopen a cleanly-closed file with a fresh writer and keep appending. Every
/// other append test holds a single writer open for all appends; this exercises
/// the persisted-state round-trip through a second `open()` (re-deriving the
/// committed dimension and EA count from disk) and confirms the second writer
/// continues at the correct ordinal rather than re-appending or gapping.
#[test]
fn recover_and_reappend_after_clean_phase4() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("reopen.h5");
    pure_create(&path, 10);

    // Writer 1: append across inline -> direct -> super, clean close.
    {
        let w = File::open_swmr_writer(&path).unwrap();
        w.dataset("d")
            .unwrap()
            .append(&(10..300).collect::<Vec<_>>())
            .unwrap();
        w.close().unwrap();
    }
    // Writer 2: re-derive committed state from disk, continue appending.
    {
        let w = File::open_swmr_writer(&path).unwrap();
        w.dataset("d")
            .unwrap()
            .append(&(300..900).collect::<Vec<_>>())
            .unwrap();
        w.close().unwrap();
    }

    let expected: Vec<i32> = (0..900).collect();
    assert_eq!(
        read_pure(&path),
        expected,
        "hdf5-pure read mismatch after reopen"
    );
    assert_eq!(
        read_c(&path),
        expected,
        "C-library read mismatch after reopen"
    );
}

/// A filtered (compressed) dataset is not an appendable target: the Extensible
/// Array stores a different element encoding and the appended chunk would have
/// to be compressed. The filter check runs at the first append (not at
/// `open()`), so `open()` succeeds, the append is rejected with a "filtered"
/// reason, and the dataset's existing (compressed) data is left readable.
#[test]
fn rejects_filtered_pure_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("filtered.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..100).collect::<Vec<_>>())
            .with_shape(&[100])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[10])
            .with_deflate(4);
        b.write(&path).unwrap();
    }

    {
        // open() only validates the superblock; the filter is caught at append.
        let w = File::open_swmr_writer(&path).unwrap();
        let err = match w
            .dataset("d")
            .unwrap()
            .append(&(100..110).collect::<Vec<_>>())
        {
            Ok(()) => panic!("expected a filtered dataset to be rejected"),
            Err(e) => e,
        };
        match err {
            Error::SwmrAppendUnsupported(reason) => {
                assert!(reason.contains("filtered"), "unexpected reason: {reason}");
            }
            other => panic!("expected SwmrAppendUnsupported, got {other:?}"),
        }
    }

    // The rejected append did not corrupt the compressed dataset.
    assert_eq!(
        read_pure(&path),
        (0..100).collect::<Vec<_>>(),
        "pure read after rejected append"
    );
    assert_eq!(
        read_c(&path),
        (0..100).collect::<Vec<_>>(),
        "C read after rejected append"
    );
}

/// The refusal above holds for a filtered dataset sitting on a *partial*
/// trailing chunk too. `File::open_rw` grows that shape in place since issue
/// #393 by re-encoding the trailing chunk and repointing its index element; a
/// SWMR writer must not, because its readers follow the header counts rather
/// than the dataspace dimension and would cross the repoint.
#[test]
fn rejects_filtered_pure_dataset_with_a_partial_trailing_chunk() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("filtered_partial.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..105).collect::<Vec<_>>())
            .with_shape(&[105]) // 105 % 10 != 0: a partial trailing chunk
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[10])
            .with_deflate(4);
        b.write(&path).unwrap();
    }

    {
        let w = File::open_swmr_writer(&path).unwrap();
        let err = w
            .dataset("d")
            .unwrap()
            .append(&(105..115).collect::<Vec<_>>())
            .expect_err("a SWMR writer must not re-encode a visible filtered chunk");
        match err {
            Error::SwmrAppendUnsupported(reason) => {
                assert!(reason.contains("filtered"), "unexpected reason: {reason}");
            }
            other => panic!("expected SwmrAppendUnsupported, got {other:?}"),
        }
    }

    assert_eq!(read_pure(&path), (0..105).collect::<Vec<_>>());
    assert_eq!(read_c(&path), (0..105).collect::<Vec<_>>());

    // The same file, the same shape, through `File::open_rw`: accepted.
    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("d")
            .unwrap()
            .append(&(105..115).collect::<Vec<_>>())
            .unwrap();
    }
    assert_eq!(read_pure(&path), (0..115).collect::<Vec<_>>());
    assert_eq!(read_c(&path), (0..115).collect::<Vec<_>>());
}
