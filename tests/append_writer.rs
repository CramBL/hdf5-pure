//! Pure-Rust tests for in-place appends through an owned handle: general appends
//! to an existing
//! chunked, unlimited, Extensible-Array-indexed dataset — filtered and
//! unfiltered, chunk-aligned and not, across one or many calls and sessions —
//! read back with this crate. C-library interop lives in
//! `crates/crosscheck/tests/append_writer.rs`.
use hdf5_pure::{Error, File, FileAccessProperties, FileBuilder, ScaleOffset, SyncPolicy};
use tempfile::tempdir;

/// Create a rank-1, unlimited i32 dataset with the given chunk length and
/// (optional) filters, seeded with `0..n`.
fn create_i32(
    path: &std::path::Path,
    n: i32,
    chunk: u64,
    deflate: bool,
    shuffle: bool,
    fletcher32: bool,
) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    let ds = b
        .create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    if shuffle {
        ds.with_shuffle();
    }
    if deflate {
        ds.with_deflate(6);
    }
    if fletcher32 {
        ds.with_fletcher32();
    }
    b.write(path).unwrap();
}

fn read_i32(path: &std::path::Path) -> Vec<i32> {
    let f = File::open(path).unwrap();
    f.dataset("d").unwrap().read_i32().unwrap()
}

/// Run `f` against a read-write file scoped so its exclusive lock is released
/// before the closure returns to the caller (so a subsequent read can reopen the
/// file — the lock is mandatory on Windows).
fn with_writer<T>(path: &std::path::Path, f: impl FnOnce(&File) -> T) -> T {
    // `SyncPolicy::OnClose`: every test here asserts what the file contains, not
    // when it reached the platter, and the default costs one `fsync` per append
    // — the dominant cost of the session loops below. The policies write
    // byte-identical files (`tests/sync_policy.rs` asserts that directly), and
    // dropping the handle still issues the closing barrier.
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let out = f(&file);
    drop(file);
    out
}

// ---- unfiltered --------------------------------------------------------------

#[test]
fn unfiltered_chunk_aligned() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 8, 4, false, false, false); // 2 full chunks
    with_writer(&path, |w| {
        w.dataset("d")
            .unwrap()
            .append(&(8..16).collect::<Vec<_>>())
            .unwrap();
    });
    assert_eq!(read_i32(&path), (0..16).collect::<Vec<_>>());
}

#[test]
fn unfiltered_unaligned_partial_tail() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // 6 elements, chunk 4 => 1 full chunk + a 2-of-4 partial on disk.
    create_i32(&path, 6, 4, false, false, false);
    with_writer(&path, |w| {
        w.dataset("d")
            .unwrap()
            .append(&(6..11).collect::<Vec<_>>())
            .unwrap();
    });
    assert_eq!(read_i32(&path), (0..11).collect::<Vec<_>>());
}

// ---- filtered ----------------------------------------------------------------

#[test]
fn deflate_shuffle_chunk_aligned() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 8, 4, true, true, false);
    with_writer(&path, |w| {
        w.dataset("d")
            .unwrap()
            .append(&(8..16).collect::<Vec<_>>())
            .unwrap();
    });
    assert_eq!(read_i32(&path), (0..16).collect::<Vec<_>>());
}

#[test]
fn fletcher32_chunk_aligned() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // Filtered appends must be chunk-aligned: start at 8 (2 full chunks of 4),
    // append 8 more (two whole chunks).
    create_i32(&path, 8, 4, true, false, true);
    with_writer(&path, |w| {
        w.dataset("d")
            .unwrap()
            .append(&(8..16).collect::<Vec<_>>())
            .unwrap();
    });
    assert_eq!(read_i32(&path), (0..16).collect::<Vec<_>>());
}

#[test]
fn scale_offset_f64_chunk_aligned() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let init: Vec<f64> = (0..8).map(|i| i as f64 * 0.25).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_f64_data(&init)
        .with_shape(&[8])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4])
        .with_scale_offset(ScaleOffset::FloatDScale(2));
    b.write(&path).unwrap();

    // 8 -> 16, two whole chunks (filtered => chunk-aligned).
    let more: Vec<f64> = (8..16).map(|i| i as f64 * 0.25).collect();
    with_writer(&path, |w| w.dataset("d").unwrap().append(&more).unwrap());

    let back = File::open(&path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_f64()
        .unwrap();
    let expected: Vec<f64> = (0..16).map(|i| i as f64 * 0.25).collect();
    assert_eq!(back.len(), expected.len());
    for (a, e) in back.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-9, "{a} != {e}");
    }
}

#[test]
fn filtered_onto_a_partial_trailing_chunk() {
    // A filtered dataset whose on-disk length is not a chunk multiple grows like
    // any other: the trailing chunk is decoded, extended, re-encoded into a
    // fresh allocation, and its index element repointed (issue #393). Across
    // separate sessions, so each append starts from what the last one left.
    let dir = tempdir().unwrap();
    let p = dir.path().join("b.h5");
    create_i32(&p, 6, 4, true, true, false); // 6 of a chunk of 4
    with_writer(&p, |w| {
        w.dataset("d").unwrap().append(&[6, 7, 8, 9]).unwrap();
    });
    assert_eq!(read_i32(&p), (0..10).collect::<Vec<_>>());
    with_writer(&p, |w| {
        w.dataset("d").unwrap().append(&[10]).unwrap();
    });
    assert_eq!(read_i32(&p), (0..11).collect::<Vec<_>>());
}

#[test]
fn filtered_unaligned_length_onto_an_aligned_dataset() {
    // An unaligned appended *length* only makes the last chunk this append
    // writes a partial one, and that chunk's index element is a fresh insert
    // past the old dimension — invisible until the dimension is published, like
    // every whole chunk beside it. The next append then grows that partial
    // chunk, which is the other half of the pair.
    let dir = tempdir().unwrap();
    let p = dir.path().join("a.h5");
    create_i32(&p, 8, 4, true, true, false);
    with_writer(&p, |w| {
        w.dataset("d").unwrap().append(&[8, 9, 10, 11, 12]).unwrap();
    });
    assert_eq!(read_i32(&p), (0..13).collect::<Vec<_>>());

    with_writer(&p, |w| {
        w.dataset("d").unwrap().append(&[13, 14, 15]).unwrap();
    });
    assert_eq!(read_i32(&p), (0..16).collect::<Vec<_>>());
}

// ---- streaming: many appends in one session ---------------------------------

#[test]
fn many_small_appends_one_session() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // chunk 1 (the streaming layout): every append is chunk-aligned, so this is
    // the fast insert-only path. Cross the inline -> direct -> super-block
    // boundary in one long session.
    create_i32(&path, 50, 1, true, true, false);
    with_writer(&path, |w| {
        // One handle for the whole session, as the owned API intends — resolving
        // it per append would re-read the header each time.
        let mut ds = w.dataset("d").unwrap();
        for v in 50..260 {
            ds.append(&[v]).unwrap();
        }
    });
    assert_eq!(read_i32(&path), (0..260).collect::<Vec<_>>());
}

#[test]
fn repeated_partial_appends_one_session() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // Start with a pure partial chunk (3 of 4), then repeatedly append amounts
    // that keep landing mid-chunk, forcing repeated in-place tail relocation.
    // Unfiltered, so the trailing element (a single address) is repointed
    // atomically each time; any-length appends are allowed.
    create_i32(&path, 3, 4, false, false, false);
    with_writer(&path, |w| {
        let mut ds = w.dataset("d").unwrap();
        ds.append(&[3, 4]).unwrap(); // -> 5
        ds.append(&[5, 6, 7]).unwrap(); // -> 8 (fills a chunk)
        ds.append(&[8]).unwrap(); // -> 9 (new partial)
        ds.append(&(9..30).collect::<Vec<_>>()).unwrap(); // -> 30
    });
    assert_eq!(read_i32(&path), (0..30).collect::<Vec<_>>());
}

// ---- reopen across sessions --------------------------------------------------

#[test]
fn reopen_and_continue() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // Unfiltered so the any-length (partial-tail) reopen path is exercised.
    create_i32(&path, 6, 4, false, false, false);
    with_writer(&path, |w| {
        w.dataset("d").unwrap().append(&[6, 7, 8, 9]).unwrap()
    }); // -> 10
    assert_eq!(read_i32(&path), (0..10).collect::<Vec<_>>());
    // Fresh session must roll forward from the committed dimension.
    with_writer(&path, |w| {
        w.dataset("d")
            .unwrap()
            .append(&(10..17).collect::<Vec<_>>())
            .unwrap();
    });
    assert_eq!(read_i32(&path), (0..17).collect::<Vec<_>>());
}

// ---- multiple datasets in one writer ----------------------------------------

#[test]
fn multiple_datasets_one_writer() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let mut b = FileBuilder::new();
    // "a" is filtered and appended chunk-aligned (start 4, chunk 2, whole-chunk
    // appends); "b" is unfiltered and appended any-length.
    b.create_dataset("a")
        .with_i32_data(&[0, 1, 2, 3])
        .with_shape(&[4])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[2])
        .with_deflate(6);
    b.create_dataset("b")
        .with_i32_data(&[100, 101])
        .with_shape(&[2])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();

    with_writer(&path, |w| {
        w.dataset("a").unwrap().append(&[4, 5]).unwrap(); // filtered: whole chunk
        w.dataset("b").unwrap().append(&[102, 103, 104]).unwrap(); // unfiltered: any length
        w.dataset("a").unwrap().append(&[6, 7]).unwrap(); // filtered: whole chunk
    });

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("a").unwrap().read_i32().unwrap(),
        vec![0, 1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        f.dataset("b").unwrap().read_i32().unwrap(),
        vec![100, 101, 102, 103, 104]
    );
}

// ---- generic + raw -----------------------------------------------------------

#[test]
fn generic_and_raw_append() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 4, 4, false, false, false);
    with_writer(&path, |w| {
        w.dataset("d").unwrap().append(&[4i32, 5, 6, 7]).unwrap();
        let mut bytes = Vec::new();
        for v in 8i32..12 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        w.dataset("d").unwrap().append_raw(&bytes).unwrap();
    });
    assert_eq!(read_i32(&path), (0..12).collect::<Vec<_>>());
}

// ---- refusals ----------------------------------------------------------------

/// The immediate in-place append refuses with `AppendInPlaceUnsupported`; the
/// staged, index-rebuilding path (`Dataset::append_staged`) reports
/// `AppendUnsupported` at commit. Either is a refusal that leaves the file
/// untouched, which is what these tests pin.
fn assert_unsupported(r: Result<(), Error>) {
    assert!(
        matches!(
            r,
            Err(Error::AppendInPlaceUnsupported(_) | Error::AppendUnsupported(_))
        ),
        "expected an append refusal, got {r:?}"
    );
}

#[test]
fn refuse_contiguous() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3]);
    b.write(&path).unwrap();
    with_writer(&path, |w| {
        assert_unsupported(w.dataset("d").unwrap().append(&[4, 5]))
    });
    assert_eq!(read_i32(&path), vec![1, 2, 3]);
}

#[test]
fn refuse_fixed_not_unlimited() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3])
        .with_maxshape(&[100])
        .with_chunks(&[2]);
    b.write(&path).unwrap();
    with_writer(&path, |w| {
        assert_unsupported(w.dataset("d").unwrap().append(&[4, 5]))
    });
    assert_eq!(read_i32(&path), vec![1, 2, 3]);
}

#[test]
fn refuse_datatype_mismatch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 4, 4, false, false, false);
    with_writer(&path, |w| {
        assert_unsupported(w.dataset("d").unwrap().append(&[4.0, 5.0]))
    });
    assert_eq!(read_i32(&path), vec![0, 1, 2, 3]);
}

#[test]
fn refuse_mixed_element_types() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 4, 4, false, false, false);
    // append() with a single call cannot mix types; drive the conflict through
    // append_raw of a wrong-length buffer instead: 3 bytes is not a whole i32.
    with_writer(&path, |w| {
        assert_unsupported(w.dataset("d").unwrap().append_raw(&[1, 2, 3]))
    });
    assert_eq!(read_i32(&path), vec![0, 1, 2, 3]);
}

#[test]
fn refuse_nonexistent_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 4, 4, false, false, false);
    // Resolving the handle is where a missing dataset is reported now, rather
    // than at open time.
    with_writer(&path, |w| {
        assert!(w.dataset("missing").is_err());
    });
    assert_eq!(read_i32(&path), vec![0, 1, 2, 3]);
}

#[test]
fn append_to_pure_empty_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // This crate allocates the Extensible-Array index eagerly, so an empty
    // dataset can be grown in place from the first append. Filtered => the append
    // is chunk-aligned (two whole chunks of 4).
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[])
        .with_shape(&[0])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4])
        .with_shuffle()
        .with_deflate(6);
    b.write(&path).unwrap();
    with_writer(&path, |w| {
        w.dataset("d")
            .unwrap()
            .append(&(0..8).collect::<Vec<_>>())
            .unwrap();
    });
    assert_eq!(read_i32(&path), (0..8).collect::<Vec<_>>());
}

#[test]
fn zero_length_append_is_noop() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, 4, 4, false, false, false);
    with_writer(&path, |w| {
        w.dataset("d").unwrap().append::<i32>(&[]).unwrap();
    });
    assert_eq!(read_i32(&path), vec![0, 1, 2, 3]);
}
