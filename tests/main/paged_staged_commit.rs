//! Tests staged PAGE commits with persistence enabled or disabled.

use hdf5_pure::EditBacking;
use hdf5_pure::File;
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

#[test]
fn paged_without_persist_uses_mirrored_staged_edits() {
    let path = temp::repo_temp_path("pure_paged_staged_nopersist.h5");
    paged::write_contiguous_i32(&path, PAGE, false, 64);
    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
    file.root()
        .create_dataset("added", |b| {
            b.with_i32_data(&[1, 2, 3]);
        })
        .unwrap();
    file.commit().unwrap();
    drop(file);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("added").unwrap().read_i32().unwrap(),
        [1, 2, 3]
    );
}

const PAGE: u64 = 4096;
