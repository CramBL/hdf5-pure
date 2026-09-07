//! Bounded-memory mutation of genuine paged files (issue #173 Phase 2, B2):
//! the bounded engine grows a persisting paged file, segregating raw and
//! metadata into separate pages, and rewrites its per-page-type managers at
//! close. libhdf5 interop lives in `crates/crosscheck/tests/file_space.rs`.

use hdf5_pure::{
    Error, File, FileAccessProperties, FileBuilder, FileSpaceStrategy, MemoryStrategy,
};

/// Open with the bounded engine demanded rather than merely preferred: these
/// tests are about that engine, so a file it stops accepting must fail here
/// rather than quietly retarget the whole file at the mirror.
fn open_bounded(path: &std::path::Path) -> Result<File, hdf5_pure::Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
}

const PAGE: u64 = 4096;

use test_util::temp;

/// A fixture under the repository's gitignored `tmp/`, in a directory of its own
/// so two concurrent runs of this binary cannot collide on the name (issue #334).
fn tmp(name: &str) -> temp::TempPath {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tmp");
    std::fs::create_dir_all(&p).expect("create the repository's tmp directory");
    temp::temp_path_in(&p, name)
}

/// Build a persisting paged file with an unlimited rank-1 chunked i32 dataset `d`
/// seeded with `0..n`.
fn build_paged(path: &std::path::Path, n: i32, chunk: u64) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(PAGE);
    b.write(path).unwrap();
}

/// Assert the on-disk paged invariants after a mutation: page-aligned file and
/// EOA, and free sections that are non-overlapping and within the file.
fn assert_paged_ok(path: &std::path::Path) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        bytes.len() as u64 % PAGE,
        0,
        "file is a whole number of pages"
    );
    let f = File::open(path).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::Page));
    let info = f.file_space_info().expect("records a strategy");
    assert!(info.persist, "still persisting");
    assert_eq!(info.page_size, PAGE);
    assert_eq!(info.eoa_pre_fsm % PAGE, 0, "EOA page-aligned");
    assert_eq!(info.eoa_pre_fsm, bytes.len() as u64, "EOA == file size");
    let free = f.persisted_free_space();
    let mut sorted = free.clone();
    sorted.sort_by_key(|&(a, _)| a);
    let mut prev_end = 0u64;
    for (addr, len) in &sorted {
        assert!(*addr >= prev_end, "sections do not overlap");
        assert!(addr + len <= bytes.len() as u64, "section within the file");
        prev_end = addr + len;
    }
}

/// A persisting paged file grows through the bounded engine: appending enough rows
/// to force extensible-array index growth allocates metadata (new EA blocks) as
/// well as raw chunks, so the append exercises page segregation. Every row reads
/// back and the paged invariants hold.
#[test]
fn paged_persist_append_roundtrip() {
    let path = tmp("pure_paged_mut_roundtrip.h5");
    build_paged(&path, 64, 64); // 1 chunk, ~one raw page

    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        let extra: Vec<i32> = (64..5000).collect(); // 77 more chunks -> EA index grows, new pages
        ds.append(&extra).unwrap();
        file.close().unwrap();
        // `ds` holds an `Arc` clone of the file handle (and thus the exclusive OS
        // lock); scope both so the lock is released before the file is read back
        // (mandatory-lock reads on Windows fail otherwise).
    }

    assert_paged_ok(&path);
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
    let path = tmp("pure_paged_mut_many.h5");
    build_paged(&path, 100, 32);

    let mut next = 100i32;
    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        for _ in 0..20 {
            let batch: Vec<i32> = (next..next + 250).collect();
            ds.append(&batch).unwrap();
            next += 250;
        }
        file.close().unwrap();
        // Drop `ds` (an `Arc` clone holding the OS lock) before reading back.
    }

    assert_paged_ok(&path);
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
    let path = tmp("pure_paged_mut_large.h5");
    build_paged(&path, 256, 256);

    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        // ~1.5 MiB of raw i32 -> more than one internal 1 MiB batch.
        let extra: Vec<i32> = (256..400_000).collect();
        ds.append(&extra).unwrap();
        file.close().unwrap();
        // Drop `ds` (an `Arc` clone holding the OS lock) before reading back.
    }

    assert_paged_ok(&path);
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
    let path = tmp("pure_paged_mirror_commit.h5");
    build_paged(&path, 100, 32);

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

    assert_paged_ok(&path);
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
    let path = tmp("pure_paged_mut_drop.h5");
    build_paged(&path, 64, 64);
    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(64..3000).collect::<Vec<i32>>()).unwrap();
        // Drop without close: the Drop guard runs finalize_persist best-effort.
    }
    assert_paged_ok(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..3000).collect::<Vec<i32>>()
    );
    assert!(
        !f.persisted_free_space().is_empty(),
        "a dropped paged handle finalizes like close"
    );
}

/// A paged file without persisted free space cannot be mutated at all, so a paged
/// file must persist its free space to be grown. `File::open_rw` says so at open;
/// the refusals further in still exist and still leave the file untouched, and are
/// reached here through `MemoryStrategy::Mirrored`.
#[test]
fn paged_non_persist_mirror_is_refused() {
    let path = tmp("pure_paged_nonpersist_mirror.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..100).collect::<Vec<i32>>())
        .with_shape(&[100])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[32]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, false, 0)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();

    // `File::open_rw` refuses this file outright: neither backing can edit it, so
    // there is nothing to gain by letting an edit be staged against it first.
    assert!(
        matches!(File::open_rw(&path), Err(Error::EditUnsupported(_))),
        "open_rw must refuse a paged non-persisting file up front"
    );

    // Demanding the mirror still opens it — that asks for a backing rather than
    // expressing a preference, and reading such a file is legitimate — which is
    // the path that still reaches the commit-time refusal below.
    {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Mirrored),
        )
        .unwrap();
        let mut ds = file.dataset("d").unwrap();
        // Immediate in-place append is refused (would break page alignment).
        assert!(
            matches!(
                ds.append(&[100i32, 101]),
                Err(Error::AppendInPlaceUnsupported(_))
            ),
            "in-place append on a paged file must be refused"
        );
        // Staged append + commit is refused too, before any writes.
        ds.append_staged(|bb| {
            bb.append_i32(&[100, 101, 102]);
        })
        .unwrap();
        assert!(
            matches!(file.commit(), Err(Error::EditUnsupported(_))),
            "commit on a paged file must be refused"
        );
        // Drop `file` and its `ds` clone (holding the OS lock) before reading back.
    }

    // The file is untouched: still reads the original and stays page-aligned.
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..100).collect::<Vec<i32>>()
    );
    assert_eq!(std::fs::read(&path).unwrap().len() as u64 % PAGE, 0);
}

/// A bounded session that appends nothing must not grow or re-page the file.
#[test]
fn paged_persist_noop_close_does_not_grow() {
    let path = tmp("pure_paged_mut_noop.h5");
    build_paged(&path, 200, 50);
    let before = std::fs::metadata(&path).unwrap().len();
    open_bounded(&path).unwrap().close().unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        before,
        "a no-append close must not grow a paged file"
    );
    assert_paged_ok(&path);
}
