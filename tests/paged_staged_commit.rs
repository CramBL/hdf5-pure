//! Staged edits on genuine paged files through `File::open_rw` (issue #198,
//! step 1). The whole-file editor used to refuse every paged file and send the
//! caller to the bounded engine; it now commits a persisting paged file
//! through a page-aware tail that keeps the per-page-type managers intact.
//! A paged file *without* persisted managers is still refused, because nothing
//! on disk records which pages hold metadata and which hold raw data.

use hdf5_pure::{
    Error, File, FileAccessProperties, FileBuilder, FileSpaceStrategy, MemoryStrategy,
};

const PAGE: u64 = 4096;

use test_util::temp;

// The page-homogeneity check is shared with the free-space tests, which exercise
// the same invariant from the in-place append side (issue #387).
use paged::assert_pages_homogeneous;
use test_util::paged;

/// A fixture under the repository's gitignored `tmp/`, in a directory of its own
/// so two concurrent runs of this binary cannot collide on the name (issue #334).
fn tmp(name: &str) -> temp::TempPath {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tmp");
    std::fs::create_dir_all(&p).expect("create the repository's tmp directory");
    temp::temp_path_in(&p, name)
}

/// Build a paged file with one contiguous i32 dataset `d` seeded with `0..n`.
fn build_paged(path: &std::path::Path, n: i32, persist: bool) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, persist, 0)
        .with_file_space_page_size(PAGE);
    b.write(path).unwrap();
}

/// The on-disk paged invariants: a whole number of pages, a page-aligned EOA
/// that matches the file length, and free sections that neither overlap nor run
/// past end-of-file.
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
    let mut free = f.persisted_free_space();
    free.sort_by_key(|&(a, _)| a);
    let mut prev_end = 0u64;
    for (addr, len) in &free {
        assert!(*addr >= prev_end, "sections do not overlap");
        assert!(addr + len <= bytes.len() as u64, "section within the file");
        prev_end = addr + len;
    }
}

/// A persisting paged file accepts a staged dataset addition through
/// `File::open_rw`: both the old and the new dataset read back, and the file
/// still satisfies every paged invariant.
#[test]
fn paged_persist_staged_create_dataset() {
    let path = tmp("pure_paged_staged_create.h5");
    build_paged(&path, 64, true);

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("added", |b| {
                b.with_i32_data(&(1000..1100).collect::<Vec<i32>>());
            })
            .unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..64).collect::<Vec<i32>>()
    );
    assert_eq!(
        f.dataset("added").unwrap().read_i32().unwrap(),
        (1000..1100).collect::<Vec<i32>>()
    );
    drop(f);
    assert_paged_ok(&path);
}

/// A staged commit that allocates both page types keeps them in separate pages.
/// The append writes raw chunks and rebuilds the chunk index (raw, since an index
/// travels with the data it indexes), and the new dataset adds raw values and a
/// metadata object header, so the commit has to switch page type several times.
#[test]
fn paged_staged_commit_keeps_pages_homogeneous() {
    let path = tmp("pure_paged_staged_homogeneous.h5");
    // A chunked, unlimited dataset so the staged append rebuilds an index.
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<i32>>())
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(PAGE);
        b.write(&path).unwrap();
    }
    {
        let s = File::open_rw(&path).unwrap();
        let mut ds = s.dataset("d").unwrap();
        ds.append_staged(|b| {
            b.append_i32(&(64..4000).collect::<Vec<i32>>());
        })
        .unwrap();
        s.root()
            .create_dataset("added", |b| {
                b.with_f64_data(&vec![2.5f64; 1024]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..4000).collect::<Vec<i32>>()
    );
    drop(f);
    assert_paged_ok(&path);
    assert_pages_homogeneous(&path, PAGE, &["d", "added"]);
}

/// A paged file with a userblock is refused rather than silently un-paged.
///
/// The session's free-space bookkeeping is not base-address aware, so it declines
/// to seed persistence on a userblock file. Committing anyway would append without
/// page awareness — mixing metadata and raw data in the file's pages and leaving
/// the end of allocation unaligned — producing a file that still advertises the
/// paged strategy but no longer satisfies it. The refusal is the same one a paged
/// non-persisting file gets, since in both cases the page bookkeeping is missing.
///
/// It is refused at *open*, and the ordering that makes it so is the point of this
/// test: a userblock is a bounded-only limitation, so `MemoryStrategy::Auto` would
/// fall back to the mirror for it — skipping this refusal on the way past — unless
/// the refusals both backings share are checked first.
#[test]
fn paged_with_userblock_is_refused() {
    let path = tmp("pure_paged_staged_userblock.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<i32>>())
            .with_shape(&[64]);
        // A paged file's userblock must be a whole number of pages, so page
        // boundaries measured from the base coincide with absolute ones.
        b.with_userblock(PAGE);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(PAGE);
        b.write(&path).unwrap();
    }

    let err = File::open_rw(&path).unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("persisted free space")),
        "expected a paged refusal for a userblock file, got {err:?}"
    );

    // Demanding the mirror still opens it, and the commit-time guard behind the
    // open-time one still fires.
    let s = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Mirrored),
    )
    .unwrap();
    s.root()
        .create_dataset("added", |b| {
            b.with_i32_data(&[1i32, 2, 3]);
        })
        .unwrap();
    let err = s.commit().unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("persisted free space")),
        "expected a paged refusal for a userblock file, got {err:?}"
    );
    drop(s);

    // The refusal happened before any byte moved: the file still reads.
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..64).collect::<Vec<i32>>()
    );
}

/// A paged file that does not persist its free space is still refused: without
/// on-disk managers there is no record of which pages are metadata and which are
/// raw, so a commit could not keep the two segregated.
///
/// `File::open_rw` now says so at open, since neither backing can edit such a
/// file. The commit-time refusal behind it still exists and is reached here by
/// demanding the mirror, which opens the file because reading it is legitimate.
#[test]
fn paged_without_persist_is_refused() {
    let path = tmp("pure_paged_staged_nopersist.h5");
    build_paged(&path, 64, false);

    let err = File::open_rw(&path).unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("persisted free space")),
        "expected a persisted-free-space refusal at open, got {err:?}"
    );

    let s = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Mirrored),
    )
    .unwrap();
    s.root()
        .create_dataset("added", |b| {
            b.with_i32_data(&[1i32, 2, 3]);
        })
        .unwrap();
    let err = s.commit().unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("persisted free space")),
        "expected a persisted-free-space refusal, got {err:?}"
    );
}
