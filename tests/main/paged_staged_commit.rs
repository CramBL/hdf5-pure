//! Staged edits on genuine paged files through `File::open_rw` (issue #198,
//! step 1). The whole-file editor used to refuse every paged file and send the
//! caller to the bounded engine; it now commits a persisting paged file
//! through a page-aware tail that keeps the per-page-type managers intact.
//! A paged file *without* persisted managers is still refused, because nothing
//! on disk records which pages hold metadata and which hold raw data.

use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::MemoryStrategy;
use test_util::temp;
use test_util_hdf5::paged;

/// A persisting paged file accepts a staged dataset addition through
/// `File::open_rw`: both the old and the new dataset read back, and the file
/// still satisfies every paged invariant.
#[test]
fn paged_persist_staged_create_dataset() {
    let path = temp::repo_temp_path("pure_paged_staged_create.h5");
    paged::write_contiguous_i32(&path, PAGE, true, 64);

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
    paged::assert_consistent(&path, PAGE);
}

/// A staged commit that allocates both page types keeps them in separate pages.
/// The append writes raw chunks and rebuilds the chunk index (raw, since an index
/// travels with the data it indexes), and the new dataset adds raw values and a
/// metadata object header, so the commit has to switch page type several times.
#[test]
fn paged_staged_commit_keeps_pages_homogeneous() {
    let path = temp::repo_temp_path("pure_paged_staged_homogeneous.h5");
    // A chunked, unlimited dataset so the staged append rebuilds an index.
    paged::write_unlimited_i32(&path, PAGE, true, 64, 64);
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
    paged::assert_consistent(&path, PAGE);
    paged::assert_pages_homogeneous(&path, PAGE, &["d", "added"]);
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
    let path = temp::repo_temp_path("pure_paged_staged_nopersist.h5");
    paged::write_contiguous_i32(&path, PAGE, false, 64);

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

const PAGE: u64 = 4096;
