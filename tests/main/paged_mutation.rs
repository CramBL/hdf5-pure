//! Bounded-memory mutation of genuine paged files (issue #173 Phase 2, B2):
//! the bounded engine grows a persisting paged file, segregating raw and
//! metadata into separate pages, and rewrites its per-page-type managers at
//! close. libhdf5 interop lives in `crates/crosscheck/tests/main/file_space.rs`.

use hdf5_pure::{Error, File, FileAccessProperties, MemoryStrategy};
use test_util::temp;
use test_util_hdf5::{paged, session};

/// A persisting paged file grows through the bounded engine: appending enough rows
/// to force extensible-array index growth allocates metadata (new EA blocks) as
/// well as raw chunks, so the append exercises page segregation. Every row reads
/// back and the paged invariants hold.
#[test]
fn paged_persist_append_roundtrip() {
    let path = temp::repo_temp_path("pure_paged_mut_roundtrip.h5");
    paged::write_unlimited_i32(&path, PAGE, true, 64, 64); // 1 chunk, ~one raw page

    {
        let file = session::open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        let extra: Vec<i32> = (64..5000).collect(); // 77 more chunks -> EA index grows, new pages
        ds.append(&extra).unwrap();
        file.close().unwrap();
        // `ds` holds an `Arc` clone of the file handle (and thus the exclusive OS
        // lock); scope both so the lock is released before the file is read back
        // (mandatory-lock reads on Windows fail otherwise).
    }

    paged::assert_consistent(&path, PAGE);
    let f = File::open(&path).unwrap();
    let got = f.dataset("d").unwrap().read_i32().unwrap();
    let want: Vec<i32> = (0..5000).collect();
    assert_eq!(got, want);
    // Growth allocated new pages beyond the original file.
    assert!(std::fs::metadata(&path).unwrap().len() > PAGE * 2);
}

/// Many separate append calls in one bounded session, each landing at a fresh
/// page-aligned raw run, then a single finalize at close.
#[test]
fn paged_persist_many_appends_one_finalize() {
    let path = temp::repo_temp_path("pure_paged_mut_many.h5");
    paged::write_unlimited_i32(&path, PAGE, true, 100, 32);

    let mut next = 100i32;
    {
        let file = session::open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        for _ in 0..20 {
            let batch: Vec<i32> = (next..next + 250).collect();
            ds.append(&batch).unwrap();
            next += 250;
        }
        file.close().unwrap();
        // Drop `ds` (an `Arc` clone holding the OS lock) before reading back.
    }

    paged::assert_consistent(&path, PAGE);
    let f = File::open(&path).unwrap();
    let got = f.dataset("d").unwrap().read_i32().unwrap();
    let want: Vec<i32> = (0..next).collect();
    assert_eq!(got, want, "all appended rows present");
}

/// A larger append (past the internal batch size) drives multiple
/// raw-then-metadata batches, so several metadata->raw page switches (and their
/// padding) occur in one call. Everything still reads back.
#[test]
fn paged_persist_large_append_multi_batch() {
    let path = temp::repo_temp_path("pure_paged_mut_large.h5");
    paged::write_unlimited_i32(&path, PAGE, true, 256, 256);

    {
        let file = session::open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        // ~1.5 MiB of raw i32 -> more than one internal 1 MiB batch.
        let extra: Vec<i32> = (256..400_000).collect();
        ds.append(&extra).unwrap();
        file.close().unwrap();
        // Drop `ds` (an `Arc` clone holding the OS lock) before reading back.
    }

    paged::assert_consistent(&path, PAGE);
    let f = File::open(&path).unwrap();
    let got = f.dataset("d").unwrap().read_i32().unwrap();
    assert_eq!(got.len(), 400_000);
    assert_eq!(got[399_999], 399_999);
    assert_eq!(got[0], 0);
}

/// The whole-file editor (`File::open_rw` + staged append + `commit`) grows a
/// persisting paged file too (issue #198): the commit takes a page-aware tail
/// that rewrites the per-page-type managers and keeps the allocation page-aligned,
/// so the appended rows read back and every paged invariant still holds. It used
/// to refuse such a file and send the caller to the bounded engine.
#[test]
fn paged_mirror_commit_appends() {
    let path = temp::repo_temp_path("pure_paged_mirror_commit.h5");
    paged::write_unlimited_i32(&path, PAGE, true, 100, 32);

    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append_staged(|b| {
            b.append_i32(&[100, 101, 102]);
        })
        .unwrap();
        file.commit().unwrap();
        // Drop `file` and its `ds` clone (holding the OS lock) before reading back.
    }

    paged::assert_consistent(&path, PAGE);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..103).collect::<Vec<i32>>()
    );
}

/// A dropped paged session (no `close`, an unclean exit) still leaves every
/// appended row durable and readable, and the Drop guard rewrites the per-page-
/// type managers into canonical shape, just like an explicit `close`.
#[test]
fn paged_persist_drop_finalizes() {
    let path = temp::repo_temp_path("pure_paged_mut_drop.h5");
    paged::write_unlimited_i32(&path, PAGE, true, 64, 64);
    {
        let file = session::open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(64..3000).collect::<Vec<i32>>()).unwrap();
        // Drop without close: the Drop guard runs finalize_persist best-effort.
    }
    paged::assert_consistent(&path, PAGE);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..3000).collect::<Vec<i32>>()
    );
    assert!(
        f.persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections()
            .section_count()
            > 0,
        "a dropped paged handle finalizes like close"
    );
}

#[test]
fn paged_non_persist_mirror_supports_staged_append() {
    let path = temp::repo_temp_path("pure_paged_nonpersist_mirror.h5");
    paged::write_unlimited_i32(&path, PAGE, false, 100, 32);
    {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Mirrored),
        )
        .unwrap();
        let mut dataset = file.dataset("d").unwrap();
        let error = dataset.append(&[100i32, 101]).unwrap_err();
        let Error::AppendInPlaceUnsupported(reason) = error else {
            panic!("expected AppendInPlaceUnsupported, got {error:?}");
        };
        assert_eq!(
            reason,
            "in-place append is not supported on a paged file \
            (H5F_FSPACE_STRATEGY_PAGE) without persisted free space; recreate the \
            file with with_file_space_settings(FileSpaceSettings::new(\
            FileSpaceStrategy::Page).with_persistence(true))"
        );
        dataset
            .append_staged(|builder| {
                builder.append_i32(&[100, 101, 102]);
            })
            .unwrap();
        file.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..103).collect::<Vec<i32>>()
    );
    assert_eq!(file.superblock().eof_address % PAGE, 0);
}

/// A bounded session that appends nothing must not grow or re-page the file.
#[test]
fn paged_persist_noop_close_does_not_grow() {
    let path = temp::repo_temp_path("pure_paged_mut_noop.h5");
    paged::write_unlimited_i32(&path, PAGE, true, 200, 50);
    let before = std::fs::metadata(&path).unwrap().len();
    session::open_bounded(&path).unwrap().close().unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        before,
        "a no-append close must not grow a paged file"
    );
    paged::assert_consistent(&path, PAGE);
}

const PAGE: u64 = 4096;
