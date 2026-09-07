//! Free-space reuse and truncation during in-place editing (issue #21).
//!
//! `File::open_rw` records the regions a commit vacates — deleted objects' blocks
//! and superseded group headers — and, within the same session, reuses them for
//! later writes instead of growing the file, truncating the file when a freed
//! run reaches end-of-file. These tests pin down both the size behavior and that
//! survivors stay byte-exact and the file stays valid.

use hdf5_pure::{
    AttrValue, EditBacking, File, FileAccessProperties, FileBuilder, FileSpaceStrategy,
    MemoryStrategy, SyncPolicy,
};

use temp::temp_path;
use test_util::temp;

// Shared with `tests/paged_staged_commit.rs`, which holds the staged commit to
// the same invariant this holds the in-place append's reserve to (issue #387).
use paged::assert_pages_homogeneous;
use test_util::paged;

/// The superblock's end-of-file must equal the actual file length after every
/// commit, including ones that truncate.
fn assert_eof_matches_file(path: &std::path::Path) {
    let file = File::open(path).unwrap();
    let eof = file.file_size();
    let actual = std::fs::metadata(path).unwrap().len();
    assert_eq!(
        eof, actual,
        "superblock EOF must match the physical file size"
    );
}

#[test]
fn delete_then_truncate_shrinks_within_session() {
    let path = temp_path("hdf5_pure_fs_shrink.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    b.write(&path).unwrap();
    let size_start = std::fs::metadata(&path).unwrap().len();

    // A single session: add a large dataset, then delete it. The deleted blocks
    // and the superseded root header form a run reaching end-of-file, so the
    // file is truncated back down rather than left bloated.
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("big", |b| {
                b.with_f64_data(&vec![7.0; 1024]);
            })
            .unwrap();
        s.commit().unwrap();
        let size_after_add = std::fs::metadata(&path).unwrap().len();
        assert!(size_after_add > size_start, "adding should grow the file");

        s.root().delete("big").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after_delete < size_after_add,
            "deleting the just-added dataset should shrink the file (was {size_after_add}, now {size_after_delete})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert!(file.dataset("big").is_err());
}

#[test]
fn churn_within_session_stays_bounded() {
    let path = temp_path("hdf5_pure_fs_churn.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let mut high_water = 0u64;
    {
        let s = File::open_rw(&path).unwrap();
        // Repeatedly add then delete a sizable dataset in the same session. With
        // reuse + truncation the file must not grow without bound across cycles.
        for i in 0..8 {
            s.root()
                .create_dataset("scratch", |b| {
                    b.with_f64_data(&vec![i as f64; 512]);
                })
                .unwrap();
            s.commit().unwrap();
            high_water = high_water.max(std::fs::metadata(&path).unwrap().len());
            s.root().delete("scratch").unwrap();
            s.commit().unwrap();
        }
    }

    let final_size = std::fs::metadata(&path).unwrap().len();
    // After the last delete the scratch space is reclaimed, so the file is far
    // smaller than the running peak — proof the freed space was reused, not
    // leaked on every cycle.
    assert!(
        final_size < high_water,
        "churn should reclaim space (peak {high_water}, final {final_size})"
    );

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn reuse_keeps_survivors_byte_exact() {
    let path = temp_path("hdf5_pure_fs_reuse_exact.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[10, 20, 30]);
    b.create_dataset("b").with_f64_data(&[1.5, 2.5]);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        // Delete b, then add c in a later commit: c's bytes should land in the
        // region b vacated. a and the newly written c must both read back exact.
        s.root().delete("b").unwrap();
        s.commit().unwrap();
        s.root()
            .create_dataset("c", |b| {
                b.with_i32_data(&[7, 8, 9, 10]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(names, vec!["a".to_string(), "c".to_string()]);
    assert_eq!(
        file.dataset("a").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
    assert_eq!(
        file.dataset("c").unwrap().read_i32().unwrap(),
        vec![7, 8, 9, 10]
    );
    assert!(file.dataset("b").is_err());
}

#[test]
fn delete_subtree_reclaims_all_members() {
    let path = temp_path("hdf5_pure_fs_subtree.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1]);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().create_group("grp").unwrap();
        s.root()
            .create_dataset("grp/x", |b| {
                b.with_f64_data(&vec![1.0; 256]);
            })
            .unwrap();
        s.root()
            .create_dataset("grp/y", |b| {
                b.with_f64_data(&vec![2.0; 256]);
            })
            .unwrap();
        s.commit().unwrap();
        let with_group = std::fs::metadata(&path).unwrap().len();

        s.root().delete("grp").unwrap();
        s.commit().unwrap();
        let after = std::fs::metadata(&path).unwrap().len();
        // The whole subtree (group header + both datasets' headers and data) is
        // reclaimed, shrinking the file well below its size with the group.
        assert!(
            after < with_group,
            "deleting a subtree should reclaim its members (was {with_group}, now {after})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert!(file.group("grp").is_err());
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), vec![1]);
}

#[test]
fn trailing_slack_past_recorded_eof_stays_readable() {
    // `commit` makes the superblock recording the smaller end-of-file durable
    // *before* it physically `set_len`s the file, so a crash in that window leaves
    // a durable, smaller superblock EOF plus the not-yet-removed trailing bytes.
    // This pins down the reader-side property that makes such a crash harmless:
    // the reader navigates by the superblock's end-of-file address and never reads
    // the slack past it. It reproduces that on-disk state by re-appending leftover
    // bytes to a cleanly committed file and confirms the file still reads exactly.
    // (It exercises the *outcome* of the ordering, not the ordering itself —
    // fault-injecting between the superblock sync and `set_len` would need a seam
    // File::open_rw does not yet expose, and remains future work.)
    let path = temp_path("hdf5_pure_fs_trailing_slack.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[11, 22, 33]);
    b.write(&path).unwrap();

    // Add then delete a large dataset so the second commit truncates the file
    // back down, leaving the recorded end-of-file equal to the physical size.
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("scratch", |b| {
                b.with_f64_data(&vec![5.0; 2048]);
            })
            .unwrap();
        s.commit().unwrap();
        s.root().delete("scratch").unwrap();
        s.commit().unwrap();
    }

    let (logical_eof, physical) = {
        let f = File::open(&path).unwrap();
        (f.superblock().eof_address, f.file_size())
    };
    assert_eq!(
        logical_eof, physical,
        "a clean truncating commit leaves no slack past the recorded end-of-file"
    );

    // Simulate the crash: the smaller-EOF superblock is already durable, but the
    // process died before `set_len`, so the freed tail is still on disk. Re-append
    // leftover bytes to reproduce that physical state.
    const SLACK: u64 = 4096;
    {
        use std::io::Write;
        let mut handle = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        handle.write_all(&vec![0xAB; SLACK as usize]).unwrap();
        handle.flush().unwrap();
    }

    // The trailing slack is invisible to the reader: survivors read byte-exact and
    // the deleted object stays gone, even though the physical file now exceeds the
    // recorded end-of-file.
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.superblock().eof_address,
        logical_eof,
        "the durable end-of-file address is unaffected by trailing slack"
    );
    assert_eq!(
        f.file_size(),
        physical + SLACK,
        "the physical file carries the leftover bytes the crash left behind"
    );
    assert_eq!(f.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![11, 22, 33]
    );
    assert!(f.dataset("scratch").is_err());
}

#[test]
fn deleting_filtered_chunked_dataset_reclaims_storage() {
    // A chunked, filtered (Fixed Array index) dataset's storage — chunk data
    // blocks plus the FAHD/FADB index — is now reclaimed on delete. Deleting it
    // shrinks the file below its size with the dataset present, the survivor
    // stays byte-exact, and the file stays valid.
    let path = temp_path("hdf5_pure_fs_chunked_filtered.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[42, 43, 44]);
    b.write(&path).unwrap();
    let size_keep_only = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("comp", |b| {
                b.with_f64_data(&vec![3.0; 4096])
                    .with_chunks(&[512])
                    .with_deflate(6);
            })
            .unwrap();
        s.commit().unwrap();
        let size_with_comp = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_with_comp > size_keep_only,
            "adding a chunked dataset should grow the file"
        );

        s.root().delete("comp").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after_delete < size_with_comp,
            "deleting the chunked dataset should reclaim its storage \
             (was {size_with_comp}, now {size_after_delete})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![42, 43, 44]
    );
    assert!(file.dataset("comp").is_err());
}

#[test]
fn deleting_unfiltered_chunked_dataset_truncates_fully() {
    // An unfiltered chunked dataset is laid out as one contiguous blob (chunk
    // data followed by the Fixed Array index, nothing between). Adding it at
    // end-of-file then deleting it in the same session reclaims the whole blob as
    // a trailing run, truncating the file back to essentially its prior size.
    let path = temp_path("hdf5_pure_fs_chunked_unfiltered.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();
    let size_start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        // 8 chunks of 512 f64 = 4096 bytes each.
        s.root()
            .create_dataset("big", |b| {
                b.with_f64_data(&vec![7.0; 4096]).with_chunks(&[512]);
            })
            .unwrap();
        s.commit().unwrap();
        let size_with_big = std::fs::metadata(&path).unwrap().len();
        assert!(size_with_big > size_start + 4096 * 8);

        s.root().delete("big").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        // The reclaimed blob formed a trailing run, so the file is truncated back
        // close to its starting size (a small, bounded amount of reused/rewritten
        // header churn aside).
        assert!(
            size_after_delete < size_start + 4096,
            "deleting an unfiltered chunked dataset should truncate the file back \
             near its start (start {size_start}, after delete {size_after_delete})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(file.dataset("big").is_err());
}

#[test]
fn deleting_paged_fixed_array_dataset_reclaims_storage() {
    // More than 1024 chunks puts the Fixed Array data block into its *paged*
    // layout (a page-init bitmap and per-page checksums). 1100 chunks of 16 f64
    // exercise that index-sizing path end to end: the whole index plus chunk
    // data is reclaimed on delete.
    let path = temp_path("hdf5_pure_fs_paged_fa.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[7]);
    b.write(&path).unwrap();
    let size_start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("paged", |b| {
                b.with_f64_data(&vec![1.0; 1100 * 16]).with_chunks(&[16]);
            })
            .unwrap(); // 1100 chunks > 1024 page size
        s.commit().unwrap();
        let size_with_paged = std::fs::metadata(&path).unwrap().len();

        s.root().delete("paged").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after_delete < size_with_paged,
            "deleting a paged fixed-array dataset should reclaim its storage \
             (was {size_with_paged}, now {size_after_delete})"
        );
        // The chunks and the index form one trailing run, so the file truncates
        // back near its start.
        assert!(size_after_delete < size_start + 4096);
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), vec![7]);
    assert!(file.dataset("paged").is_err());
}

#[test]
fn deleting_extensible_dataset_reclaims_storage() {
    // A dataset with an unlimited maximum dimension uses an Extensible Array chunk
    // index (EAHD/EAIB and, past the inline slots, data blocks). Deleting it
    // reclaims the whole index plus its chunk data.
    let path = temp_path("hdf5_pure_fs_extensible.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[9]);
    b.write(&path).unwrap();
    let size_start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        // 64 chunks of 64 f64 — enough to spill past the index block's inline
        // slots into separate data blocks, exercising the data-block walk.
        s.root()
            .create_dataset("ext", |b| {
                b.with_f64_data(&vec![2.5; 4096])
                    .with_chunks(&[64])
                    .with_maxshape(&[u64::MAX]);
            })
            .unwrap();
        s.commit().unwrap();
        let size_with_ext = std::fs::metadata(&path).unwrap().len();
        assert!(size_with_ext > size_start);

        s.root().delete("ext").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after_delete < size_with_ext,
            "deleting an extensible-array dataset should reclaim its storage \
             (was {size_with_ext}, now {size_after_delete})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), vec![9]);
    assert!(file.dataset("ext").is_err());
}

#[test]
fn deleting_single_chunk_dataset_reclaims_storage() {
    // A dataset whose single chunk covers the whole shape uses the single-chunk
    // index (the chunk address lives in the layout message, no separate index
    // structure). Deleting it reclaims that one chunk block.
    let path = temp_path("hdf5_pure_fs_single_chunk.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[5, 6]);
    b.write(&path).unwrap();
    let size_start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        // 1024 f64 = 8192 bytes in a single chunk, whose blob reclaims as one
        // trailing run.
        s.root()
            .create_dataset("one", |b| {
                b.with_f64_data(&vec![1.25; 1024]).with_chunks(&[1024]);
            })
            .unwrap(); // one chunk covers the whole dataset
        s.commit().unwrap();
        let size_with_one = std::fs::metadata(&path).unwrap().len();
        assert!(size_with_one > size_start + 8192);

        s.root().delete("one").unwrap();
        s.commit().unwrap();
        let size_after_delete = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after_delete < size_with_one,
            "deleting a single-chunk dataset should reclaim its chunk \
             (was {size_with_one}, now {size_after_delete})"
        );
        // The 8192-byte chunk dominates; reclaim truncates the file back near
        // its starting size.
        assert!(size_after_delete < size_start + 4096);
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![5, 6]
    );
    assert!(file.dataset("one").is_err());
}

#[test]
fn chunked_churn_within_session_stays_bounded() {
    // Repeatedly add then delete a sizable chunked dataset in one session. With
    // chunk + index reclaim and reuse the file must not grow without bound.
    let path = temp_path("hdf5_pure_fs_chunked_churn.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let mut high_water = 0u64;
    {
        let s = File::open_rw(&path).unwrap();
        for i in 0..8 {
            s.root()
                .create_dataset("scratch", |b| {
                    b.with_f64_data(&vec![i as f64; 2048])
                        .with_chunks(&[256])
                        .with_deflate(4);
                })
                .unwrap();
            s.commit().unwrap();
            high_water = high_water.max(std::fs::metadata(&path).unwrap().len());
            s.root().delete("scratch").unwrap();
            s.commit().unwrap();
        }
    }

    let final_size = std::fs::metadata(&path).unwrap().len();
    assert!(
        final_size < high_water,
        "chunked churn should reclaim space (peak {high_water}, final {final_size})"
    );

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn deleting_subtree_with_chunked_members_reclaims() {
    // Deleting a group reclaims its chunked-dataset members' storage too (the
    // free walk descends the subtree).
    let path = temp_path("hdf5_pure_fs_chunked_subtree.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1]);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().create_group("grp").unwrap();
        s.root()
            .create_dataset("grp/a", |b| {
                b.with_f64_data(&vec![1.0; 4096]).with_chunks(&[512]);
            })
            .unwrap();
        s.root()
            .create_dataset("grp/b", |b| {
                b.with_f64_data(&vec![2.0; 2048])
                    .with_chunks(&[256])
                    .with_deflate(6);
            })
            .unwrap();
        s.commit().unwrap();
        let with_group = std::fs::metadata(&path).unwrap().len();

        s.root().delete("grp").unwrap();
        s.commit().unwrap();
        let after = std::fs::metadata(&path).unwrap().len();
        assert!(
            after < with_group,
            "deleting a subtree with chunked members should reclaim them \
             (was {with_group}, now {after})"
        );
    }

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert!(file.group("grp").is_err());
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), vec![1]);
}

#[test]
fn persisted_chunked_reclaim_is_disjoint_and_reusable() {
    // With persistence on, deleting a chunked dataset records its storage as
    // free sections that stay disjoint and are reused across reopen.
    let path = temp_path("hdf5_pure_fs_chunked_persist.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1; 50]);
    b.create_dataset("comp")
        .with_f64_data(&vec![3.0; 4096])
        .with_chunks(&[512])
        .with_deflate(6);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("comp").unwrap();
        s.commit().unwrap();
    }
    assert_eof_matches_file(&path);

    let f = File::open(&path).unwrap();
    assert!(f.file_space_info().unwrap().persist);
    let mut free = f.persisted_free_space();
    free.sort();
    for w in free.windows(2) {
        assert!(
            w[0].0 + w[0].1 <= w[1].0,
            "persisted free regions must be disjoint after a chunked delete: {free:?}"
        );
    }
    assert_eq!(f.dataset("keep").unwrap().read_i32().unwrap(), vec![1; 50]);
    assert!(f.dataset("comp").is_err());
}

#[test]
fn persisted_free_space_survives_reopen_and_is_reused() {
    // A file created with persist = true keeps its freed space recorded on disk
    // (the FSHD/FSSE managers), so a freed region survives close/reopen and a
    // later session reuses it instead of growing the file. This is the cross-
    // session counterpart to the within-session reuse above.
    let path = temp_path("hdf5_pure_fs_persist_roundtrip.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("big").with_i32_data(&[7; 400]); // 1600 bytes of raw data
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    // Session 1: delete "big"; its storage is persisted as free space.
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
    }
    assert_eof_matches_file(&path);

    // A fresh reader recovers the persisted free space across the reopen, and the
    // surviving datasets stay byte-exact alongside the on-disk managers.
    let f = File::open(&path).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::FsmAggr));
    assert!(f.file_space_info().unwrap().persist);
    let free = f.persisted_free_space();
    let total: u64 = free.iter().map(|(_, l)| l).sum();
    assert!(
        total >= 1600 && free.iter().any(|&(_, l)| l >= 1600),
        "the deleted dataset's storage is persisted as a free section: {free:?}"
    );
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    assert_eq!(f.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
    assert!(f.dataset("big").is_err());
    drop(f);

    // Session 2: opening seeds the free list from the persisted managers, so a
    // new dataset reuses the freed hole rather than appending its data at EOF.
    let eof_before = std::fs::metadata(&path).unwrap().len();
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("d", |b| {
                b.with_i32_data(&[9; 300]);
            })
            .unwrap(); // 1200 bytes, fits the hole
        s.commit().unwrap();
    }
    assert_eof_matches_file(&path);
    let eof_after = std::fs::metadata(&path).unwrap().len();
    assert!(
        eof_after < eof_before + 1200,
        "the new dataset should reuse the persisted free hole, not grow the file \
         by its full size (before={eof_before} after={eof_after})"
    );

    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![9; 300]);
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    assert_eq!(f.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
    // Persistence remains armed: the file still records its free space on disk.
    assert!(f.file_space_info().unwrap().persist);
}

#[test]
fn persisted_managers_stay_consistent_across_many_commits() {
    // Several persisting commits in a row: each supersedes the previous on-disk
    // managers and extension, recording them as free, so the file stays valid and
    // its tracked free space never double-counts or loses a region. Both this
    // crate and a fresh reader must agree on the result after every step.
    let path = temp_path("hdf5_pure_fs_persist_multi.h5");
    let mut b = FileBuilder::new();
    for i in 0..6 {
        b.create_dataset(&format!("d{i}"))
            .with_i32_data(&vec![i; 200]);
    }
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    // Churn across multiple sessions: delete some, add some, each its own commit.
    for round in 0..4 {
        {
            let s = File::open_rw(&path).unwrap();
            s.root().delete(&format!("d{round}")).unwrap();
            s.root()
                .create_dataset(&format!("n{round}"), |b| {
                    b.with_i32_data(&vec![100 + round; 150]);
                })
                .unwrap();
            s.commit().unwrap();
        } // release the editor's lock before reading the file back
        // Free regions never overlap and the file remains valid after each round.
        assert_eof_matches_file(&path);
        let f = File::open(&path).unwrap();
        assert!(f.file_space_info().unwrap().persist);
        let mut free = f.persisted_free_space();
        free.sort();
        for w in free.windows(2) {
            assert!(
                w[0].0 + w[0].1 <= w[1].0,
                "persisted free regions must be disjoint: {free:?}"
            );
        }
    }

    // Every surviving dataset reads back byte-exact.
    let f = File::open(&path).unwrap();
    for i in 4..6 {
        assert_eq!(
            f.dataset(&format!("d{i}")).unwrap().read_i32().unwrap(),
            vec![i; 200]
        );
    }
    for round in 0..4 {
        assert_eq!(
            f.dataset(&format!("n{round}")).unwrap().read_i32().unwrap(),
            vec![100 + round; 150]
        );
    }
    assert!(f.dataset("d0").is_err());
}

/// The bytes a chunked dataset of `elems` `f64` elements occupies, near enough:
/// the chunk data itself, which dominates its index and header.
fn f64_data_bytes(elems: usize) -> u64 {
    (elems * 8) as u64
}

#[test]
fn chunked_dataset_reuses_the_hole_a_chunked_delete_left() {
    // The headline of issue #261. A chunked dataset's data region — chunk bytes
    // plus the index that addresses them — used to be *appended* unconditionally,
    // because its embedded addresses were computed from the end-of-file it was
    // about to land at. Sizing it before placing it lets it go into a freed region
    // instead, so delete-then-write of the same shape costs the file nothing.
    //
    // The hole is interior on purpose: `tail` sits above the deleted dataset, so
    // truncation cannot be what keeps the file small.
    let path = temp_path("hdf5_pure_fs_chunked_reuse.h5");
    const ELEMS: usize = 32768; // 256 KiB of raw data
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("big")
        .with_f64_data(&vec![1.0; ELEMS])
        .with_chunks(&[4096]);
    b.create_dataset("tail").with_i32_data(&[9; 16]);
    b.write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
        s.root()
            .create_dataset("big2", |b| {
                b.with_f64_data(&vec![2.0; ELEMS]).with_chunks(&[4096]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let end = std::fs::metadata(&path).unwrap().len();
    assert!(
        end < start + f64_data_bytes(ELEMS) / 10,
        "the replacement dataset should land in the hole the deleted one left, \
         not past it (start={start}, end={end})"
    );

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("big2").unwrap().read_f64().unwrap(),
        vec![2.0; ELEMS]
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(f.dataset("tail").unwrap().read_i32().unwrap(), vec![9; 16]);
    assert!(f.dataset("big").is_err());
}

#[test]
fn filtered_chunked_dataset_reuses_freed_space() {
    // Same as above for a *filtered* dataset, whose compressed size is not known
    // until the pipeline has run: the placement is chosen from the compressed
    // set's size, so the filter pass still happens exactly once.
    let path = temp_path("hdf5_pure_fs_chunked_reuse_filtered.h5");
    const ELEMS: usize = 32768;
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("big")
        .with_f64_data(&(0..ELEMS).map(|i| i as f64).collect::<Vec<f64>>())
        .with_chunks(&[4096])
        .with_deflate(4);
    b.create_dataset("tail").with_i32_data(&[9; 16]);
    b.write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();
    let hole = start; // an upper bound on what the deleted dataset can free

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
        let after_delete = std::fs::metadata(&path).unwrap().len();
        s.root()
            .create_dataset("big2", |b| {
                b.with_f64_data(&(0..ELEMS).map(|i| i as f64).collect::<Vec<f64>>())
                    .with_chunks(&[4096])
                    .with_deflate(4);
            })
            .unwrap();
        s.commit().unwrap();
        let end = std::fs::metadata(&path).unwrap().len();
        assert!(
            end < after_delete + hole / 10,
            "an identically filtered replacement should reuse the freed region \
             (after_delete={after_delete}, end={end})"
        );
    }

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("big2").unwrap().read_f64().unwrap(),
        (0..ELEMS).map(|i| i as f64).collect::<Vec<f64>>()
    );
    assert_eq!(f.dataset("tail").unwrap().read_i32().unwrap(), vec![9; 16]);
}

#[test]
fn reusing_a_chunked_hole_keeps_its_neighbors_byte_exact() {
    // Reuse writes over a dead interior region, so the test that matters is what
    // happens to the *live* bytes on either side of it. Both neighbors — one
    // below the hole, one above — must read back exactly, and every chunk of the
    // dataset written into the hole must too.
    let path = temp_path("hdf5_pure_fs_chunked_reuse_neighbors.h5");
    const ELEMS: usize = 16384;
    let below: Vec<f64> = (0..2048).map(|i| i as f64 * 0.5).collect();
    let above: Vec<i32> = (0..2048).map(|i| i * 3).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("below").with_f64_data(&below);
    b.create_dataset("victim")
        .with_f64_data(&vec![7.0; ELEMS])
        .with_chunks(&[2048]);
    b.create_dataset("above").with_i32_data(&above);
    b.write(&path).unwrap();

    let replacement: Vec<f64> = (0..ELEMS).map(|i| (i % 97) as f64).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("victim").unwrap();
        s.commit().unwrap();
        s.root()
            .create_dataset("fresh", |b| {
                b.with_f64_data(&replacement).with_chunks(&[2048]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("below").unwrap().read_f64().unwrap(), below);
    assert_eq!(f.dataset("above").unwrap().read_i32().unwrap(), above);
    assert_eq!(f.dataset("fresh").unwrap().read_f64().unwrap(), replacement);
}

#[test]
fn paged_commit_reuses_freed_space_within_its_page_type() {
    // A paged file segregates metadata and raw data by page, so it tracks free
    // space per page type and a commit may only draw from the list matching what
    // it is placing. It used to draw from none of them and always append.
    //
    // The file stays valid and every page stays homogeneous — the crosscheck
    // suite reads these files with the reference C library, which is where a
    // mixed page would show up.
    let path = temp_path("hdf5_pure_fs_paged_reuse.h5");
    const ELEMS: usize = 32768;
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("big")
        .with_f64_data(&vec![1.0; ELEMS])
        .with_chunks(&[4096]);
    b.create_dataset("tail").with_i32_data(&[9; 16]);
    b.write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
        s.root()
            .create_dataset("big2", |b| {
                b.with_f64_data(&vec![2.0; ELEMS]).with_chunks(&[4096]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let end = std::fs::metadata(&path).unwrap().len();
    assert!(
        end < start + f64_data_bytes(ELEMS) / 4,
        "a paged file should reuse its freed raw pages rather than append past \
         them (start={start}, end={end})"
    );

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("big2").unwrap().read_f64().unwrap(),
        vec![2.0; ELEMS]
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(f.dataset("tail").unwrap().read_i32().unwrap(), vec![9; 16]);
}

#[test]
fn persisted_chunked_free_space_is_reused_after_a_reopen() {
    // The cross-session case: one session deletes a chunked dataset, a *later*
    // one writes a fresh one. The second session only knows about the hole from
    // the on-disk managers it seeds its free list from, so this pins the seeding
    // and the chunked placement together.
    let path = temp_path("hdf5_pure_fs_chunked_persist_reuse.h5");
    const ELEMS: usize = 32768;
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("big")
        .with_f64_data(&vec![1.0; ELEMS])
        .with_chunks(&[4096]);
    b.create_dataset("tail").with_i32_data(&[9; 16]);
    b.write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
    }
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("big2", |b| {
                b.with_f64_data(&vec![2.0; ELEMS]).with_chunks(&[4096]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let end = std::fs::metadata(&path).unwrap().len();
    assert!(
        end < start + f64_data_bytes(ELEMS) / 10,
        "the second session should reuse the hole the first one persisted \
         (start={start}, end={end})"
    );

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("big2").unwrap().read_f64().unwrap(),
        vec![2.0; ELEMS]
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn both_read_write_backings_reuse_a_freed_hole_alike() {
    // One engine drives two backings (issue #198): the bounded one writes through
    // a handle, the mirrored one through a whole-file buffer. `File::open_rw`
    // prefers bounded, so every test above exercises that half; this one pins the
    // other, since a hole is reused by writing into the middle of the image and
    // that is precisely where the two backings differ.
    const ELEMS: usize = 16384;
    for (name, strategy) in [
        ("hdf5_pure_fs_backing_bounded.h5", MemoryStrategy::Bounded),
        ("hdf5_pure_fs_backing_mirrored.h5", MemoryStrategy::Mirrored),
    ] {
        let path = temp_path(name);
        let mut b = FileBuilder::new();
        b.create_dataset("big")
            .with_f64_data(&vec![1.0; ELEMS])
            .with_chunks(&[2048]);
        b.create_dataset("tail").with_i32_data(&[9; 16]);
        b.write(&path).unwrap();
        let start = std::fs::metadata(&path).unwrap().len();

        {
            let s = File::open_rw_with_options(
                &path,
                FileAccessProperties::new().with_memory_strategy(strategy),
            )
            .unwrap();
            assert_eq!(
                s.edit_backing(),
                Some(match strategy {
                    MemoryStrategy::Bounded => EditBacking::Bounded,
                    _ => EditBacking::Mirrored,
                }),
                "{name}: the requested backing is the one under test"
            );
            s.root().delete("big").unwrap();
            s.commit().unwrap();
            s.root()
                .create_dataset("big2", |b| {
                    b.with_f64_data(&vec![2.0; ELEMS]).with_chunks(&[2048]);
                })
                .unwrap();
            s.commit().unwrap();
        }

        let end = std::fs::metadata(&path).unwrap().len();
        assert!(
            end < start + f64_data_bytes(ELEMS) / 10,
            "{name}: the replacement should reuse the freed hole \
             (start={start}, end={end})"
        );
        assert_eof_matches_file(&path);
        let f = File::open(&path).unwrap();
        assert_eq!(
            f.dataset("big2").unwrap().read_f64().unwrap(),
            vec![2.0; ELEMS]
        );
        assert_eq!(f.dataset("tail").unwrap().read_i32().unwrap(), vec![9; 16]);
    }
}

#[test]
fn churn_of_groups_attributes_and_datasets_stays_bounded() {
    // The whole free-space story on one file: groups carrying attributes, plain
    // and filtered chunked datasets, deleted by subtree and rewritten identically,
    // round after round. Every round frees exactly what the next one needs, so a
    // file that reuses its free space returns to the same size each time while one
    // that appends grows by a full round's worth per cycle.
    //
    // A "ceiling" dataset written above the first round is what makes this a test
    // of *reuse*. Without it every freed round would reach end-of-file and be
    // truncated away, which keeps the file just as small while reusing nothing.
    let path = temp_path("hdf5_pure_fs_full_churn.h5");
    const ROUNDS: usize = 5;
    const GROUPS: usize = 4;
    const ELEMS: usize = 8192; // 64 KiB per dataset

    let write_round = |s: &File, round: usize| {
        for g in 0..GROUPS {
            let name = format!("g{g}");
            s.root()
                .create_group_with(&name, |grp| {
                    for a in 0..8 {
                        grp.set_attr(
                            &format!("attr{a}"),
                            AttrValue::F64Array(vec![round as f64; 32]),
                        );
                    }
                })
                .unwrap();
            s.root()
                .create_dataset(&format!("{name}/plain"), |b| {
                    b.with_f64_data(&vec![round as f64; ELEMS])
                        .with_chunks(&[1024]);
                })
                .unwrap();
            s.root()
                .create_dataset(&format!("{name}/filtered"), |b| {
                    b.with_f64_data(&(0..ELEMS).map(|i| (i + round) as f64).collect::<Vec<f64>>())
                        .with_chunks(&[1024])
                        .with_deflate(4);
                })
                .unwrap();
        }
    };

    let mut b = FileBuilder::new();
    b.create_dataset("anchor").with_i32_data(&[7; 64]);
    b.write(&path).unwrap();

    let baseline;
    let mut peak;
    {
        let s = File::open_rw(&path).unwrap();
        write_round(&s, 0);
        s.commit().unwrap();
        // Everything churned from here on sits *below* this dataset, so a freed
        // round is an interior hole rather than a trailing run.
        s.root()
            .create_dataset("ceiling", |b| {
                b.with_f64_data(&vec![1.25; ELEMS]);
            })
            .unwrap();
        s.commit().unwrap();
        baseline = std::fs::metadata(&path).unwrap().len();
        peak = baseline;

        for round in 1..=ROUNDS {
            for g in 0..GROUPS {
                s.root().delete(&format!("g{g}")).unwrap();
            }
            s.commit().unwrap();
            write_round(&s, round);
            s.commit().unwrap();
            peak = peak.max(std::fs::metadata(&path).unwrap().len());
        }
    }

    let end = std::fs::metadata(&path).unwrap().len();
    assert!(
        end < baseline + baseline / 4,
        "churning {ROUNDS} identical rounds should reuse the space each round \
         frees, not accumulate it (baseline={baseline}, peak={peak}, final={end})"
    );

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("anchor").unwrap().read_i32().unwrap(),
        vec![7; 64]
    );
    for g in 0..GROUPS {
        assert_eq!(
            f.dataset(&format!("g{g}/plain"))
                .unwrap()
                .read_f64()
                .unwrap(),
            vec![ROUNDS as f64; ELEMS]
        );
        assert_eq!(
            f.dataset(&format!("g{g}/filtered"))
                .unwrap()
                .read_f64()
                .unwrap(),
            (0..ELEMS)
                .map(|i| (i + ROUNDS) as f64)
                .collect::<Vec<f64>>()
        );
        let attrs = f.group(&format!("g{g}")).unwrap().attrs().unwrap();
        for a in 0..8 {
            assert_eq!(
                attrs.get(&format!("attr{a}")),
                Some(&AttrValue::F64Array(vec![ROUNDS as f64; 32])),
            );
        }
    }
}

#[test]
fn corrupt_persisted_section_is_skipped_not_fatal() {
    // A malformed free-space manager must not be trusted blindly: if a persisted
    // section claims a region past end-of-file, seeding it and later handing it
    // out would write out of bounds. The editor skips such a section instead, so
    // the open + commit still succeeds and the live data stays intact.
    let path = temp_path("hdf5_pure_fs_persist_corrupt.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[5; 50]);
    b.create_dataset("victim").with_i32_data(&[6; 300]); // 1200-byte data block
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("victim").unwrap(); // frees the largest tracked section
        s.commit().unwrap();
    }

    // Find the largest persisted free section, then corrupt its offset to an
    // out-of-bounds address. FSSE groups sections by size, each group being a
    // `count`, a `size`, then `count` `(offset, class)` pairs; locate the largest
    // section's `size` field and rewrite the offset that follows it. (Counts and
    // sizes are 1 and 8 bytes here.)
    let big_len = File::open(&path)
        .unwrap()
        .persisted_free_space()
        .into_iter()
        .map(|(_, len)| len)
        .max()
        .expect("the deletion left a persisted free section");
    let mut bytes = std::fs::read(&path).unwrap();
    let file_len = bytes.len() as u64;
    let fsse = bytes
        .windows(4)
        .position(|w| w == b"FSSE")
        .expect("a persisted file with a freed section has an FSSE block");
    let size_le = big_len.to_le_bytes();
    let size_pos = fsse
        + 13
        + bytes[fsse + 13..]
            .windows(8)
            .position(|w| w == size_le)
            .expect("the largest section's size field is in the FSSE block");
    let off = size_pos + 8; // the offset field immediately follows the size
    bytes[off..off + 8].copy_from_slice(&(file_len + 4096).to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    // Add a dataset large enough that only the (corrupt) largest region could
    // satisfy it: without the bounds guard, the allocator would pick that region
    // and `write_at` an out-of-bounds address, panicking. The guard skips it, so
    // the data is appended instead and the commit succeeds.
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("added", |b| {
                b.with_i32_data(&[9; 250]);
            })
            .unwrap(); // 1000 bytes
        s.commit().unwrap();
    }
    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("keep").unwrap().read_i32().unwrap(), vec![5; 50]);
    assert_eq!(
        f.dataset("added").unwrap().read_i32().unwrap(),
        vec![9; 250]
    );
}

/// A paged file under delete-and-recreate churn stops growing rather than
/// spending a page per commit (issue #286).
///
/// Every commit on a persisting paged file rewrites the superblock extension and
/// the free-space manager blocks. Those used to be appended at a fresh
/// page-aligned end-of-file and padded out to a page boundary, with the padding
/// recorded nowhere: a workload that stayed within a fixed budget by deleting its
/// oldest objects still grew without bound, at roughly one page per commit, and
/// `reusable_free_bytes` accounted for a fraction of what the file had spent.
///
/// The assertion is on the *shape* of the growth rather than a byte figure. The
/// file is allowed to reach whatever size the live data and its own layout need
/// over the first few rounds; what it may not do is keep climbing once the
/// workload is in its steady state, which is what an unreclaimed page per commit
/// looks like from the outside.
#[test]
fn paged_churn_reaches_a_steady_size() {
    const PAGE: u64 = 16384;
    const ROUNDS: usize = 12;
    const LIVE: usize = 2;

    let path = temp_path("hdf5_pure_fs_paged_churn.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("seed").with_i32_data(&[0i32; 4]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();

    let rows: Vec<i32> = (0..500).map(|i| i % 97).collect();
    let mut sizes = Vec::new();
    for round in 0..ROUNDS {
        // A fresh session per round, since the defect's worst half only shows
        // across a close: free space is handed back to disk in a manager that
        // records no page type, and until this fix a reopened session could never
        // spend it on metadata again.
        {
            let f = File::open_rw(&path).unwrap();
            f.root().create_group(&format!("g{round}")).unwrap();
            f.commit().unwrap();
            let g = f.group(&format!("g{round}")).unwrap();
            for name in ["a", "b"] {
                g.create_dataset(name, |b| {
                    b.with_i32_data(&rows)
                        .with_shape(&[rows.len() as u64])
                        .with_maxshape(&[u64::MAX])
                        .with_chunks(&[128]);
                })
                .unwrap();
            }
            f.commit().unwrap();
        }
        if round >= LIVE {
            let f = File::open_rw(&path).unwrap();
            f.root().delete(&format!("g{}", round - LIVE)).unwrap();
            f.commit().unwrap();
            // Every byte the file has spent and released is offered back, rather
            // than a fraction of it: the padding around the rewritten managers
            // used to be recorded nowhere at all. This also pins that the tail
            // reused rather than appended this round — an appended tail leaves the
            // remainder of the page it opened in the session's list only, where the
            // managers on disk cannot yet name it.
            let acct = f.space_accounting().unwrap();
            drop(f);
            let persisted: u64 = File::open(&path)
                .unwrap()
                .persisted_free_space()
                .iter()
                .map(|(_, len)| len)
                .sum();
            assert_eq!(
                acct.reusable_free_bytes, persisted,
                "round {round}: free space the session can spend must be the free \
                 space the file records"
            );
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }
    }

    // The steady state: the last third of the run adds nothing. Deleting one
    // round's group makes room for the next round's, and the commit tail lands in
    // the space the previous tail vacated.
    let settled = sizes[sizes.len() * 2 / 3];
    let last = *sizes.last().unwrap();
    assert_eq!(
        last, settled,
        "a paged file under steady churn must stop growing (sizes by round: {sizes:?})"
    );

    let f = File::open(&path).unwrap();
    for round in ROUNDS - LIVE..ROUNDS {
        let g = f.group(&format!("g{round}")).unwrap();
        for name in ["a", "b"] {
            assert_eq!(g.dataset(name).unwrap().read_i32().unwrap(), rows);
        }
    }
    drop(f);
    assert_eof_matches_file(&path);
}

/// A paged commit's tail — the rewritten extension and manager blocks — is
/// placed in space the file already had free, instead of opening a page for
/// itself at end-of-file.
///
/// The size test above states the consequence; this states the mechanism, which a
/// size assertion alone would not distinguish from the file simply having room to
/// spare. The tail does not settle at one fixed address — best fit picks whatever
/// hole suits as the free list shuffles — so what is asserted is that whatever
/// address it takes was free *before* the commit that wrote it, which is exactly
/// what it never was while the tail could only append.
#[test]
fn a_paged_commit_tail_is_placed_in_free_space() {
    const PAGE: u64 = 16384;
    let path = temp_path("hdf5_pure_fs_paged_tail_reuse.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("seed").with_i32_data(&[0i32; 4]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();
    let start_len = std::fs::metadata(&path).unwrap().len();

    // Three commits that change nothing but one attribute and the tail the commit
    // has to rewrite regardless.
    for i in 0..3 {
        let before: Vec<(u64, u64)> = File::open(&path).unwrap().persisted_free_space();
        let f = File::open_rw(&path).unwrap();
        f.root().set_attr("n", AttrValue::I64(i)).unwrap();
        f.commit().unwrap();
        drop(f);

        let f = File::open(&path).unwrap();
        let managers: Vec<u64> = f
            .file_space_info()
            .expect("a persisting file records its managers")
            .manager_addrs
            .iter()
            .copied()
            .filter(|&a| a != u64::MAX)
            .collect();
        assert!(
            !managers.is_empty(),
            "commit {i}: the file has free space, so it has managers to place"
        );
        for addr in managers {
            assert!(
                before.iter().any(|&(a, len)| addr >= a && addr < a + len),
                "commit {i}: a manager block landed at {addr}, which was not free \
                 before the commit ({before:?}) — the tail appended instead of \
                 reusing"
            );
        }
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            start_len,
            "commit {i}: rewriting the tail must not grow the file"
        );
        // Once the tail is reusing rather than appending, a commit that rewrites
        // only the tail neither gains nor loses free space: the new tail takes
        // exactly what the old one gives back. A commit that took more room than
        // it filled would show up here as free space quietly draining away, which
        // is the same leak in miniature.
        if i > 0 {
            let before_total: u64 = before.iter().map(|(_, len)| len).sum();
            let after_total: u64 = f.persisted_free_space().iter().map(|(_, len)| len).sum();
            assert_eq!(
                after_total, before_total,
                "commit {i}: rewriting the tail must conserve free space"
            );
        }
    }
    assert_eof_matches_file(&path);
}

/// Across a sweep of layouts, a paged commit that rewrites only its tail
/// conserves free space exactly.
///
/// The tail is sized against the free lists it then draws from, so its length and
/// the hole it lands in determine each other, and the arithmetic settles
/// differently depending on how the file happens to be laid out — whether the
/// chosen hole is consumed whole or merely shrunk, whether the remainder changes
/// which manager records it. A single fixture exercises one of those settlements.
/// Sweeping the filler size walks the tail through many, and the invariant is the
/// same in all of them: what the new tail takes is what the old one gave back,
/// to the byte. A tail that reserved more room than it filled would leave the
/// difference recorded by nobody, and the total would drop.
#[test]
fn a_paged_tail_conserves_free_space_across_layouts() {
    const PAGE: u64 = 16384;
    for filler in 0..64usize {
        let path = temp_path(&format!("hdf5_pure_fs_paged_sweep_{filler}.h5"));
        let mut b = FileBuilder::new();
        b.create_dataset("seed")
            .with_i32_data(&(0..100 + filler as i32).collect::<Vec<i32>>());
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
            .with_file_space_page_size(PAGE);
        b.write(&path).unwrap();

        // The first commit settles the tail into the file's own free space; from
        // the second on, each tail replaces the previous one and nothing else
        // moves.
        //
        // `OnClose` for the same reason as the flat sweep below: what these
        // sessions are read for is the free space each one leaves recorded, and
        // the ordering `Always` buys with two or three `fsync`s a commit is not
        // something any assertion here looks at.
        let mut totals = Vec::new();
        for i in 0..3 {
            let f = File::open_rw_with_options(
                &path,
                FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
            )
            .unwrap();
            f.root().set_attr("n", AttrValue::I64(i)).unwrap();
            f.commit().unwrap();
            drop(f);
            totals.push(
                File::open(&path)
                    .unwrap()
                    .persisted_free_space()
                    .iter()
                    .map(|(_, len)| len)
                    .sum::<u64>(),
            );
        }
        assert_eq!(
            totals[1], totals[2],
            "filler {filler}: rewriting the tail must conserve free space, not \
             quietly retire some of it (totals {totals:?})"
        );
    }
}

/// A **non-paged** persisting file's commit tail — the rewritten extension and
/// its one `FSHD`/`FSSE` pair — is placed in space an earlier commit freed, the
/// same way the paged tail above is (issue #358).
///
/// The paged tail learned this first (issue #286); the flat one kept appending,
/// so every commit grew the file by a tail and freed its predecessor's, and a
/// `FileSpaceStrategy::FsmAggr` file under churn climbed forever while reporting
/// all of it as reusable. As above, the address is not fixed — best fit picks
/// whatever hole suits — so what is asserted is that the manager landed where the
/// file was already free, which is what it never was while the tail could only
/// append.
#[test]
fn a_persisting_commit_tail_is_placed_in_free_space() {
    let path = temp_path("hdf5_pure_fs_flat_tail_reuse.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("seed").with_i32_data(&[0i32; 4]);
    b.create_dataset("scratch").with_i32_data(&[7i32; 256]);
    // A live object above the hole, so the deletion below leaves an interior hole
    // rather than a run reaching end-of-file.
    b.create_dataset("ceiling").with_i32_data(&[9i32; 4]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    // The hole the tail will move into: a fresh file has none, so the first tail
    // has nowhere to go but end-of-file.
    {
        let f = File::open_rw(&path).unwrap();
        f.root().delete("scratch").unwrap();
        f.commit().unwrap();
    }
    let start_len = std::fs::metadata(&path).unwrap().len();

    // Three commits that change nothing but one attribute and the tail the commit
    // has to rewrite regardless.
    for i in 0..3 {
        let before: Vec<(u64, u64)> = File::open(&path).unwrap().persisted_free_space();
        let f = File::open_rw(&path).unwrap();
        f.root().set_attr("n", AttrValue::I64(i)).unwrap();
        f.commit().unwrap();
        drop(f);

        let f = File::open(&path).unwrap();
        let managers: Vec<u64> = f
            .file_space_info()
            .expect("a persisting file records its managers")
            .manager_addrs
            .iter()
            .copied()
            .filter(|&a| a != u64::MAX)
            .collect();
        assert!(
            !managers.is_empty(),
            "commit {i}: the file has free space, so it has a manager to place"
        );
        for addr in managers {
            assert!(
                before.iter().any(|&(a, len)| addr >= a && addr < a + len),
                "commit {i}: a manager block landed at {addr}, which was not free \
                 before the commit ({before:?}) — the tail appended instead of \
                 reusing"
            );
        }
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            start_len,
            "commit {i}: rewriting the tail must not grow the file"
        );
    }

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("seed").unwrap().read_i32().unwrap(), [0; 4]);
    assert_eq!(f.dataset("ceiling").unwrap().read_i32().unwrap(), [9; 4]);
}

/// The consequence of the test above, on the workload the issue describes: a
/// non-paged persisting file under delete-and-recreate churn stops growing
/// (issue #358).
///
/// Shaped like [`paged_churn_reaches_a_steady_size`]: the file may reach whatever
/// size its live data and layout need over the first rounds, but once the
/// workload is in its steady state it may not keep climbing. A tail appended per
/// commit is what climbing looks like from the outside — about 590 bytes a round
/// here, forever, with every one of them recorded as free.
#[test]
fn persisting_churn_reaches_a_steady_size() {
    const ROUNDS: usize = 16;
    const LIVE: usize = 2;

    let path = temp_path("hdf5_pure_fs_flat_churn.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("seed").with_i32_data(&[0i32; 4]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    let rows: Vec<i32> = (0..500).map(|i| i % 97).collect();
    let mut sizes = Vec::new();
    for round in 0..ROUNDS {
        // A fresh session per round, so each round's tail is placed from the free
        // list a reopen seeds out of the on-disk managers rather than from one the
        // session has been carrying.
        {
            let f = File::open_rw(&path).unwrap();
            f.root().create_group(&format!("g{round}")).unwrap();
            f.commit().unwrap();
            let g = f.group(&format!("g{round}")).unwrap();
            for name in ["a", "b"] {
                g.create_dataset(name, |b| {
                    b.with_i32_data(&rows);
                })
                .unwrap();
            }
            f.commit().unwrap();
        }
        if round >= LIVE {
            let f = File::open_rw(&path).unwrap();
            f.root().delete(&format!("g{}", round - LIVE)).unwrap();
            f.commit().unwrap();
            drop(f);
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }
    }

    let settled = sizes[sizes.len() * 2 / 3];
    let last = *sizes.last().unwrap();
    assert_eq!(
        last, settled,
        "a persisting file under steady churn must stop growing (sizes by round: {sizes:?})"
    );

    let f = File::open(&path).unwrap();
    for round in ROUNDS - LIVE..ROUNDS {
        let g = f.group(&format!("g{round}")).unwrap();
        for name in ["a", "b"] {
            assert_eq!(g.dataset(name).unwrap().read_i32().unwrap(), rows);
        }
    }
    drop(f);
    assert_eof_matches_file(&path);
}

/// Across a sweep of layouts, a non-paged persisting file whose commits rewrite
/// only the tail holds its size (issue #358).
///
/// The flat counterpart of [`a_paged_tail_conserves_free_space_across_layouts`],
/// asserting size rather than the free-space total, because the two say different
/// things here: the flat tail may take a few more bytes than its blocks fill, so
/// consecutive commits legitimately record totals a section record apart, while
/// the file's *size* may not move at all once a tail is landing in free space.
///
/// A single fixture reaches one settlement of an arithmetic that depends on the
/// whole free list — the tail's length and the hole it lands in determine each
/// other — so this sweeps the filler size and the file carries three holes rather
/// than one, which is what makes the settlement turn over as the tail moves. The
/// difference that buys is measured: requiring the tail to fill its reservation
/// exactly leaves `a_persisting_commit_tail_is_placed_in_free_space` passing, and
/// fails here from filler 18 on, where the file gains a tail per commit again.
#[test]
fn a_persisting_tail_holds_its_size_across_layouts() {
    for filler in 0..64usize {
        let path = temp_path(&format!("hdf5_pure_fs_flat_sweep_{filler}.h5"));
        let mut b = FileBuilder::new();
        b.create_dataset("seed")
            .with_i32_data(&(0..100 + filler as i32).collect::<Vec<i32>>());
        // Three holes of unrelated sizes rather than one, so the free list the
        // tail is sized against has several sections and its length turns over as
        // the tail moves between them.
        for (i, len) in [256usize, 37, 91].into_iter().enumerate() {
            b.create_dataset(&format!("scratch{i}"))
                .with_i32_data(&vec![7i32; len + filler]);
            b.create_dataset(&format!("keep{i}"))
                .with_i32_data(&[9i32; 4]);
        }
        b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
        b.write(&path).unwrap();

        // Nine sessions per filler, 576 across the sweep, and what each one is
        // read for is the size of the file it leaves behind. `SyncPolicy::Always`
        // would spend two or three `fsync`s per commit establishing an ordering
        // no assertion here looks at; `OnClose` keeps the one barrier that is
        // load-bearing, at the close before each size is read, and writes the
        // same bytes either way (`tests/sync_policy.rs`).
        let open = |path: &std::path::Path| {
            File::open_rw_with_options(
                path,
                FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
            )
            .unwrap()
        };

        for i in 0..3 {
            let f = open(&path);
            f.root().delete(&format!("scratch{i}")).unwrap();
            f.commit().unwrap();
        }

        // The first of these settles the tail into a hole; from the second on,
        // each commit replaces an earlier tail and nothing else moves.
        let mut sizes = Vec::new();
        for i in 0..6 {
            let f = open(&path);
            f.root().set_attr("n", AttrValue::I64(i)).unwrap();
            f.commit().unwrap();
            drop(f);
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }
        assert_eq!(
            sizes[1],
            *sizes.last().unwrap(),
            "filler {filler}: rewriting the tail must not grow the file (sizes {sizes:?})"
        );
        assert_eof_matches_file(&path);
    }
}

/// A paged file reuses a freed hole across a close, not only within one session
/// (issue #358, which reported the opposite).
///
/// `paged_commit_reuses_freed_space_within_its_page_type` deletes and re-adds in
/// one session, where the hole is in the list that session has been carrying. Here
/// the delete and the re-add are separate sessions, so the second one knows the
/// hole only from the per-page-type managers it seeds its free list from on open —
/// the paged counterpart of `persisted_chunked_free_space_is_reused_after_a_reopen`.
#[test]
fn paged_free_space_is_reused_after_a_reopen() {
    const ELEMS: usize = 32768;
    let path = temp_path("hdf5_pure_fs_paged_persist_reuse.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("big")
        .with_f64_data(&vec![1.0; ELEMS])
        .with_chunks(&[4096]);
    // Above `big`, and more than a page of it, so deleting `big` leaves an
    // interior hole rather than a run reaching end-of-file that the commit
    // releases: sixteen elements landed in a low hole and left the pages
    // trailing after all.
    b.create_dataset("tail")
        .with_f64_data(&vec![9.0f64; CEILING_ELEMS])
        .with_chunks(&[4096]);
    b.write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
    }
    {
        let s = File::open_rw(&path).unwrap();
        assert!(
            s.space_accounting().unwrap().reusable_free_bytes >= f64_data_bytes(ELEMS),
            "the reopened session must recover the freed dataset's pages"
        );
        s.root()
            .create_dataset("big2", |b| {
                b.with_f64_data(&vec![2.0; ELEMS]).with_chunks(&[4096]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let end = std::fs::metadata(&path).unwrap().len();
    assert!(
        end < start + f64_data_bytes(ELEMS) / 4,
        "a reopened paged session should write into the pages the delete freed \
         (start={start}, end={end})"
    );

    assert_eof_matches_file(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("big2").unwrap().read_f64().unwrap(),
        vec![2.0; ELEMS]
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(
        f.dataset("tail").unwrap().read_f64().unwrap(),
        vec![9.0f64; CEILING_ELEMS]
    );
}

/// Every free section a released file records lies inside it, and a session that
/// reopens the file agrees with the length the release left (issue #418).
///
/// The free-space managers name absolute file offsets, so a run given back to
/// the filesystem has to leave the record as well as the file — a section past
/// the end of the file is one the reference C library would hand a writer.
fn assert_released_file_is_self_consistent(path: &std::path::Path) {
    assert_eof_matches_file(path);
    let len = std::fs::metadata(path).unwrap().len();
    let f = File::open(path).unwrap();
    for (addr, size) in f.persisted_free_space() {
        assert!(
            addr + size <= len,
            "a persisted free section [{addr}, {}) runs past the end of the \
             {len}-byte file",
            addr + size
        );
    }
    drop(f);
    let s = File::open_rw(path).unwrap();
    assert_eq!(
        s.space_accounting().unwrap().logical_size,
        len,
        "a reopened session must account for the file at the length the release \
         left it"
    );
}

/// Stage a group of populated chunked datasets under `name` and commit it.
fn create_populated_group(f: &File, name: &str, rows: usize, cols: usize) {
    let payload: Vec<i32> = (0..rows as i32).collect();
    f.root()
        .create_group_with(name, |g| {
            for c in 0..cols {
                g.create_dataset(&format!("c{c}"), |b| {
                    b.with_i32_data(&payload)
                        .with_shape(&[rows as u64])
                        .with_chunks(&[512])
                        .with_maxshape(&[u64::MAX]);
                });
            }
        })
        .unwrap();
    f.commit().unwrap();
}

/// Deleting a populated group gives the space at the end of the file back to the
/// filesystem rather than leaving the file at its high-water mark with those
/// bytes merely recorded as free (issue #418).
///
/// What is guaranteed is the release, not that a delete never grows a file:
/// persisting the free-space managers costs metadata that has to go somewhere,
/// and the reference C library grows a file to persist them too. So this
/// measures the file against the data left alive in it — a few kilobytes of
/// manager blocks and the reserve the next commit's tail needs, not the
/// megabyte the deleted group occupied.
#[test]
fn deleting_a_populated_group_releases_the_end_of_a_flat_file() {
    const ROWS: usize = 65536; // 256 KiB per dataset
    let path = temp_path("hdf5_pure_fs_release_flat.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 0);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let f = open_rw_on_close(&path);
    create_populated_group(&f, "scratch", ROWS, 2);
    let peak = f.file_size();
    assert!(
        peak > 2 * ROWS as u64 * 4,
        "the fixture must put a megabyte or so in the file, not {peak} bytes"
    );

    f.root().delete("scratch").unwrap();
    f.commit().unwrap();
    let after = f.file_size();
    assert_eq!(
        after,
        f.space_accounting().unwrap().logical_size,
        "file_size and the session's own accounting must report one length"
    );
    // Everything the group occupied reached the end of the file, so all of it
    // goes back; what is left is the surviving objects and the managers.
    assert!(
        after < peak / 16,
        "deleting the group should have given the end of the file back (peak \
         {peak}, now {after})"
    );
    f.close().unwrap();

    assert_released_file_is_self_consistent(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    assert!(f.group("scratch").is_err(), "the group is gone");
}

/// The same on a paged file, where the end of allocation stays a whole number of
/// pages: deleting down to one small group releases the pages above it
/// (issue #418).
///
/// The reporter's first case, keeping the *first* group rather than the last —
/// with the last, the surviving group's own blocks sit at the top of the file
/// and no amount of releasing can move them, which is a question for repack
/// rather than for this.
#[test]
fn deleting_down_to_one_group_releases_a_paged_file_s_trailing_pages() {
    const ROWS: usize = 8192;
    const GROUPS: usize = 6;
    let path = temp_path("hdf5_pure_fs_release_paged.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(RECLAIM_PAGE);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let f = open_rw_on_close(&path);
    for i in 0..GROUPS {
        create_populated_group(&f, &format!("e{i:05}"), ROWS, 2);
    }
    let peak = f.file_size();
    assert!(
        peak > 16 * RECLAIM_PAGE,
        "the fixture must reach a good many pages, not {peak} bytes"
    );

    for i in 1..GROUPS {
        f.root().delete(&format!("e{i:05}")).unwrap();
        f.commit().unwrap();
    }
    let after = f.file_size();
    assert_eq!(
        after % RECLAIM_PAGE,
        0,
        "a paged file's end of allocation stays a whole number of pages ({after})"
    );
    assert!(
        after < peak / 2,
        "deleting down to one group should have released the pages above it \
         (peak {peak}, now {after})"
    );
    f.close().unwrap();

    assert_released_file_is_self_consistent(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    assert_eq!(
        f.dataset("e00000/c0").unwrap().read_i32().unwrap(),
        (0..ROWS as i32).collect::<Vec<i32>>()
    );
}

/// A release survives the close and the reopen, and the reopened session goes on
/// editing the shorter file (issue #418).
///
/// The length is the one thing a crash-atomic commit publishes *twice* — in the
/// superblock's end-of-file and in the physical file — so a release that got the
/// order wrong would leave a file whose superblock names bytes that are not
/// there. Reading it back with a fresh session is what catches that.
#[test]
fn a_released_file_reopens_and_keeps_being_edited() {
    const ROWS: usize = 65536;
    let path = temp_path("hdf5_pure_fs_release_reopen.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 0);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let peak = {
        let f = open_rw_on_close(&path);
        create_populated_group(&f, "scratch", ROWS, 2);
        let peak = f.file_size();
        f.root().delete("scratch").unwrap();
        f.commit().unwrap();
        f.close().unwrap();
        peak
    };
    let released = std::fs::metadata(&path).unwrap().len();
    assert!(
        released < peak / 16,
        "the close must leave the released length on disk (peak {peak}, now \
         {released})"
    );
    assert_released_file_is_self_consistent(&path);

    // A second round of the same churn on the reopened file settles at the same
    // length rather than starting again from the high-water mark.
    {
        let f = open_rw_on_close(&path);
        create_populated_group(&f, "scratch2", ROWS, 2);
        f.root().delete("scratch2").unwrap();
        f.commit().unwrap();
        f.close().unwrap();
    }
    let again = std::fs::metadata(&path).unwrap().len();
    assert!(
        again <= released + RECLAIM_PAGE,
        "a second round of the same churn must not leave the file larger \
         (was {released}, now {again})"
    );
    assert_released_file_is_self_consistent(&path);
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
}

/// A file that has released the end of itself then holds that length across a
/// long run of tail-rewriting commits, instead of moving between two lengths a
/// manager tail apart (issue #418).
///
/// This is what pins `TRAILING_RESERVE_TAILS` against being one. That reserve —
/// the few manager tails' worth of a released run that the commit leaves in the
/// file — is not a tidiness margin: the blocks are rewritten by *every* commit
/// and can never land in their own predecessor's extent, because the on-disk
/// superblock still points at it until the repoint. Trimmed to a single tail's
/// worth, the reserve is the size of the tail that made it, and the next tail is
/// a section longer whenever the section set has grown, so it goes past the end
/// of the file instead: at a reserve of one, this loop leaves the flat file
/// alternating between 5,992 and 6,303 bytes, so 100 of its 200 commits leave a
/// length other than the one before them. What rules out *two* is not here but in
/// `persisting_churn_reaches_a_steady_size`, which at that value has not settled
/// by its sixteenth round.
///
/// A commit with nothing staged returns before the tail rewrite, so the loop
/// overwrites one root attribute — a fixed-width `I64`, so the header it
/// rewrites is the same size every round and the only thing that can move is the
/// tail.
#[test]
fn a_released_file_holds_its_length_across_many_tail_rewrites() {
    const ROWS: usize = 100;
    const GROUPS: usize = 4;
    const COMMITS: i64 = 200;

    for (label, strategy, slack) in [
        // A tail is a few hundred bytes on a flat file and a page on a paged one,
        // which is the whole of the slack either may take.
        ("flat", FileSpaceStrategy::FsmAggr, 1024),
        ("paged", FileSpaceStrategy::Page, RECLAIM_PAGE),
    ] {
        let path = temp_path(&format!("hdf5_pure_fs_release_loop_{label}.h5"));
        let mut b = FileBuilder::new();
        b.with_file_space_strategy(strategy, true, 0);
        if strategy == FileSpaceStrategy::Page {
            b.with_file_space_page_size(RECLAIM_PAGE);
        }
        b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
        b.write(&path).unwrap();

        let f = open_rw_on_close(&path);
        for i in 0..GROUPS {
            create_populated_group(&f, &format!("e{i}"), ROWS, 2);
        }
        for i in 1..GROUPS {
            f.root().delete(&format!("e{i}")).unwrap();
            f.commit().unwrap();
        }
        // One commit past the deletes, so the length being measured is the one
        // the release settles at rather than the one the last delete published:
        // a commit whose tail had to be appended above the run it freed moves it
        // down on the commit after.
        f.root().set_attr("n", AttrValue::I64(-1)).unwrap();
        f.commit().unwrap();
        let released = f.file_size();

        let mut peak = released;
        let mut moved = 0usize;
        for i in 0..COMMITS {
            f.root().set_attr("n", AttrValue::I64(i)).unwrap();
            f.commit().unwrap();
            peak = peak.max(f.file_size());
            moved += usize::from(f.file_size() != released);
        }
        assert!(
            peak <= released + slack,
            "{label}: {COMMITS} tail rewrites must not climb past one tail's \
             worth of the released length (released {released}, peak {peak})"
        );
        assert_eq!(
            f.file_size(),
            released,
            "{label}: and must leave the file at the length the release left it"
        );
        // The load-bearing one: the length is not merely back where it started,
        // it never left. A reserve too small to take the next tail sends it past
        // end-of-file and the file moves on every commit.
        assert_eq!(
            moved, 0,
            "{label}: {moved} of {COMMITS} tail rewrites changed the file's \
             length (released {released}, peak {peak})"
        );
        f.close().unwrap();

        assert_released_file_is_self_consistent(&path);
        let f = File::open(&path).unwrap();
        assert_eq!(f.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
        assert_eq!(
            f.dataset("e0/c0").unwrap().read_i32().unwrap(),
            (0..ROWS as i32).collect::<Vec<i32>>()
        );
    }
}

/// Build `name` as a rank-1, unlimited, chunked i32 dataset of length zero.
fn create_log(session: &File, name: &str) {
    session
        .root()
        .create_dataset(name, |b| {
            b.with_i32_data(&[])
                .with_shape(&[0])
                .with_chunks(&[1024])
                .with_maxshape(&[u64::MAX]);
        })
        .unwrap();
    session.commit().unwrap();
}

/// An immediate [`Dataset::append`] must allocate its chunks out of the free
/// space a prior commit left, exactly as `create_dataset` + `commit` does
/// (issue #349). Before the fix the append always extended end-of-file, so the
/// replacement dataset doubled the file rather than moving into the hole its
/// predecessor vacated.
#[test]
fn immediate_append_reuses_a_freed_hole() {
    let payload: Vec<i32> = (0..16384).collect();
    let path = temp_path("hdf5_pure_fs_immediate_append_reuse.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    create_log(&session, "log");
    session.dataset("log").unwrap().append(&payload).unwrap();
    let after_first = std::fs::metadata(&path).unwrap().len();

    // A ceiling above the churned region: without a live object after it, the
    // deletion below leaves a free run reaching end-of-file that the commit
    // truncates away, and the test would pass on truncation rather than on reuse.
    session
        .root()
        .create_dataset("ceiling", |b| {
            b.with_i32_data(&[9, 9, 9]);
        })
        .unwrap();
    session.commit().unwrap();

    session.root().delete("log").unwrap();
    session.commit().unwrap();
    let freed = session.space_accounting().unwrap().reusable_free_bytes;
    assert!(
        freed >= payload.len() as u64 * 4,
        "the deleted dataset's chunks should be reusable (only {freed} bytes are)"
    );

    create_log(&session, "log2");
    session.dataset("log2").unwrap().append(&payload).unwrap();
    session.close().unwrap();
    let after_second = std::fs::metadata(&path).unwrap().len();

    assert!(
        after_second < after_first + freed / 2,
        "the second append should have moved into the {freed}-byte hole, not \
         extended the file (was {after_first}, now {after_second})"
    );
    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("log2").unwrap().read_i32().unwrap(), payload);
    // Reuse must not have written over either survivor.
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    assert_eq!(
        file.dataset("ceiling").unwrap().read_i32().unwrap(),
        [9, 9, 9]
    );
}

/// An immediate append onto a file that **persists** its free-space managers may
/// spend only space it has first taken *out* of them (issue #387), and takes a
/// hole out whenever the append fits in it, however far under a batch the hole
/// is (issue #413).
///
/// The managers are a durable record, and an append has no superblock repoint of
/// its own to update that record with. So the bytes it writes must already be
/// ones no manager advertises: the session draws what it can, republishes the
/// managers without it, and spends only from there. What it must never do is
/// write into a hole the published managers still offer — through a clean close
/// as much as a crash, that would leave the next session (this crate or the C
/// library) allocating over a live chunk.
///
/// The hole here is far smaller than one reserve batch. That used to mean nothing
/// was drawn and the append extended end-of-file, so a persisting file whose
/// deleted objects were all under a megabyte grew forever. Now the hole is drawn
/// and spent: the file does not grow, the session's reusable space falls by what
/// the append placed, and the managers on disk afterwards advertise nothing a
/// live chunk occupies — which is the half of this that is about safety rather
/// than size, and holds whichever way the draw is sized.
#[test]
fn a_persisting_file_reuses_a_hole_smaller_than_a_batch() {
    let payload: Vec<i32> = (0..16384).collect();
    let payload_bytes = payload.len() as u64 * 4;
    let path = temp_path("hdf5_pure_fs_immediate_append_persisting.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    // Comfortably larger than the payload, so the hole still holds the whole
    // append after the commits below have placed their own tails inside it —
    // and still well under the batch an append reserves at most, so this is a
    // hole the old floor declined.
    b.create_dataset("scratch")
        .with_i32_data(&vec![7; payload.len() + 1024]);
    b.create_dataset("ceiling").with_i32_data(&[9, 9, 9]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    session.root().delete("scratch").unwrap();
    session.commit().unwrap();
    // After the commit that creates the dataset, so the staged path's own
    // (legitimate) reuse is not counted against the append.
    create_log(&session, "log");
    let freed = session.space_accounting().unwrap().reusable_free_bytes;
    assert!(freed >= payload_bytes, "expected a sizable hole");
    let size_before = session.file_size();

    session.dataset("log").unwrap().append(&payload).unwrap();
    assert_eq!(
        session.file_size(),
        size_before,
        "the append fits in the hole, so the file must not grow"
    );
    let left = session.space_accounting().unwrap().reusable_free_bytes;
    assert!(
        left + payload_bytes <= freed,
        "the append must have spent the hole: {freed} bytes were reusable before it \
         and {left} are after, for a {payload_bytes}-byte payload"
    );
    session.close().unwrap();

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("log").unwrap().read_i32().unwrap(), payload);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    let persisted: u64 = file.persisted_free_space().iter().map(|(_, l)| l).sum();
    assert!(
        persisted + payload_bytes <= freed,
        "the managers must describe the hole less what the append placed in it \
         ({persisted} of {freed})"
    );
    drop(file);
    assert_no_persisted_free_space_holds_live_chunks(&path, &["log"]);
}

/// Elements the persisting-reuse tests churn: 2 MiB of `i32`.
///
/// Sized against the batch an append reserves out of the on-disk managers, which
/// is a megabyte: the payload has to be several of those, or the test would be
/// measuring a single draw rather than a session that keeps drawing as it
/// appends. It is also what makes the hole the delete leaves worth drawing from
/// at all.
const REUSE_ELEMS: i32 = 512 * 1024;
/// Elements per chunk in those tests: 64 KiB chunks, four pages of the paged
/// fixture's 16 KiB page.
const REUSE_CHUNK: u64 = 16384;

/// A ceiling object big enough to be placed past end-of-file rather than in one
/// of the file's holes — more than a page on a paged file — so that what a delete
/// below it leaves really is an interior hole and not a run the commit releases
/// (issue #418).
const CEILING_ELEMS: usize = 4096;

/// No region the on-disk free-space managers advertise may overlap a live chunk
/// of any of `datasets`.
///
/// This is the half of issue #387 that is about safety rather than size. A
/// manager offering space a chunk occupies is not a stale number: the next
/// session — this crate or the reference C library — allocates straight out of
/// those managers and would write over the top of live data.
fn assert_no_persisted_free_space_holds_live_chunks(path: &std::path::Path, datasets: &[&str]) {
    let file = File::open(path).unwrap();
    let free = file.persisted_free_space();
    for name in datasets {
        for chunk in file.dataset(name).unwrap().chunks().unwrap() {
            let (lo, hi) = (chunk.address, chunk.address + chunk.storage_size);
            for &(addr, len) in &free {
                assert!(
                    addr >= hi || lo >= addr + len,
                    "the managers advertise [{addr}, {}) as free, which overlaps a live \
                     chunk of `{name}` at [{lo}, {hi})",
                    addr + len
                );
            }
        }
    }
}

/// Stage an empty unlimited chunked dataset and commit it.
fn create_reuse_log(session: &File, name: &str) {
    session
        .root()
        .create_dataset(name, |b| {
            b.with_i32_data(&[])
                .with_shape(&[0])
                .with_chunks(&[REUSE_CHUNK])
                .with_maxshape(&[u64::MAX]);
        })
        .unwrap();
    session.commit().unwrap();
}

/// The reporter's scenario for issue #387: append a large payload in place,
/// delete the dataset, commit, and append the same volume onto a replacement.
///
/// Returns the file size after each wave and the reusable free space before and
/// after the second one.
fn append_churn_on_a_persisting_file(
    strategy: FileSpaceStrategy,
    name: &str,
) -> (u64, u64, u64, u64) {
    let payload: Vec<i32> = (0..REUSE_ELEMS).collect();
    let path = temp_path(name);
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(strategy, true, 1);
    if strategy == FileSpaceStrategy::Page {
        b.with_file_space_page_size(RECLAIM_PAGE);
    }
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let session = open_rw_on_close(&path);
    create_reuse_log(&session, "log");
    session.dataset("log").unwrap().append(&payload).unwrap();
    // A live object above the churned region, so the delete below leaves an
    // interior hole rather than a run reaching end-of-file that the commit would
    // simply truncate away — which would let this pass on truncation rather than
    // on reuse. Sized past every hole the file holds at this point and past a
    // page: a three-element one landed in a low hole and left the payload
    // trailing after all.
    session
        .root()
        .create_dataset("ceiling", |b| {
            b.with_i32_data(&vec![9i32; CEILING_ELEMS]);
        })
        .unwrap();
    session.commit().unwrap();
    let after_first = std::fs::metadata(&path).unwrap().len();

    session.root().delete("log").unwrap();
    session.commit().unwrap();
    let freed_before = session.space_accounting().unwrap().reusable_free_bytes;

    create_reuse_log(&session, "log2");
    session.dataset("log2").unwrap().append(&payload).unwrap();
    let freed_after = session.space_accounting().unwrap().reusable_free_bytes;
    session.close().unwrap();
    let after_second = std::fs::metadata(&path).unwrap().len();

    assert_eof_matches_file(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("log2").unwrap().read_i32().unwrap(), payload);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    assert_eq!(
        file.dataset("ceiling").unwrap().read_i32().unwrap(),
        vec![9i32; CEILING_ELEMS]
    );
    drop(file);
    assert_no_persisted_free_space_holds_live_chunks(&path, &["log2"]);

    (after_first, after_second, freed_before, freed_after)
}

/// Hold one run of [`append_churn_on_a_persisting_file`] to the reuse issue #387
/// asks for: the second wave lands inside the hole the first one left rather than
/// past it, and the reusable free space falls as it is spent.
fn assert_the_second_wave_reused_the_hole(first: u64, second: u64, before: u64, after: u64) {
    let payload_bytes = REUSE_ELEMS as u64 * 4;
    assert!(
        before >= payload_bytes,
        "the deleted dataset's chunks should be reusable (only {before} bytes are)"
    );
    assert!(
        second < first + payload_bytes / 2,
        "the second wave should have moved into the {before}-byte hole, not extended \
         the file (was {first}, now {second})"
    );
    assert!(
        after < before / 2,
        "the reusable free space should have fallen as the append spent it (was \
         {before}, now {after})"
    );
}

/// A file that persists its free-space managers reuses the hole a delete left,
/// through the in-place `Dataset::append` path (issue #387).
///
/// Before this the append always extended end-of-file on such a file, so the
/// second wave cost a whole extra copy of the payload while `reusable_free_bytes`
/// went on reporting the first copy's space as available.
#[test]
fn a_persisting_file_reuses_a_hole_it_has_taken_out_of_the_managers() {
    let (first, second, before, after) = append_churn_on_a_persisting_file(
        FileSpaceStrategy::FsmAggr,
        "hdf5_pure_fs_persisting_append_reuse.h5",
    );
    assert_the_second_wave_reused_the_hole(first, second, before, after);
}

/// The same on a paged file, where the reserve is drawn as raw-typed space so an
/// append spending it cannot put raw bytes in a metadata page (issue #387).
#[test]
fn a_paged_file_reuses_a_hole_it_has_taken_out_of_the_managers() {
    let (first, second, before, after) = append_churn_on_a_persisting_file(
        FileSpaceStrategy::Page,
        "hdf5_pure_fs_paged_append_reuse.h5",
    );
    assert_the_second_wave_reused_the_hole(first, second, before, after);
}

/// An append spending a reserve on a paged file leaves every page holding one
/// kind of byte (issue #387).
///
/// Page homogeneity is the invariant paging exists for and the one thing the
/// reference C library cannot report on — it reads a mixed file happily — so the
/// test above cannot stand in for this: reading every value back and finding the
/// managers truthful both hold perfectly well on a file whose pages have been
/// mixed. It is why the reserve is drawn through `PagedEdit::alloc_typed` rather
/// than out of the flat list.
///
/// Many chunk-sized appends rather than one large one, so the session exhausts a
/// reserve and draws again several times over, at a chunk width that is not a
/// divisor of the page — every allocation out of the reserve then starts and ends
/// mid-page, where a page-sized one would stay aligned and hide any misfiling by
/// construction. A staged commit follows, so a *metadata* allocation is made
/// while those partly-spent pages are in the lists.
///
/// What this does **not** discriminate, stated plainly rather than left to be
/// discovered: no mutation of the reserve's page typing has been found that fails
/// it. Drawing as metadata instead of raw is caught elsewhere — the draw is sized
/// on the raw list's own figure, so an allocator asked for metadata may not serve
/// it and `WriteEngine::take_raw_spans` asserts on the disagreement, or in a
/// release build draws nothing, which
/// `paged_group_churn_with_populated_datasets_reaches_a_steady_size` fails on as
/// growth — but not here, and not as a mixed page. Returning the unspent
/// remainder to the metadata list does not fail this either, because placement
/// is best-fit and a large misfiled run is the last fragment a small metadata
/// allocation would choose. So this is a net for the mixing class — a raw append
/// packing into a live metadata page, a metadata allocation landing in a raw one
/// — and not a proof of the `PageType::Raw` argument, which rests on the
/// reasoning at `WriteEngine::take_raw_span` instead.
#[test]
fn a_paged_append_reserve_keeps_pages_homogeneous() {
    // Deliberately *not* a divisor of the page: 6,000 bytes of chunk against a
    // 16,384-byte page, so the reserve is spent in pieces that leave its unspent
    // remainder starting mid-page. A page-sized chunk would keep every allocation
    // page-aligned and no misfiling of that remainder could ever show.
    const CHUNK: u64 = 1500;
    // Comfortably more than two reserve batches once appended, so the run crosses
    // several draws.
    const ROUNDS: usize = 400;

    let path = temp_path("hdf5_pure_fs_paged_reserve_homogeneous.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    b.with_file_space_page_size(RECLAIM_PAGE);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    // The hole the appends draw from: four megabytes of raw chunk data an earlier
    // commit gives back.
    b.create_dataset("victim")
        .with_i32_data(&vec![7i32; 1024 * 1024])
        .with_chunks(&[CHUNK]);
    b.create_dataset("ceiling")
        .with_i32_data(&vec![9i32; CEILING_ELEMS]);
    b.write(&path).unwrap();

    let session = open_rw_on_close(&path);
    session.root().delete("victim").unwrap();
    session.commit().unwrap();
    let freed = session.space_accounting().unwrap().reusable_free_bytes;
    assert!(
        freed > 2 * 1024 * 1024,
        "the fixture must leave more than two reserve batches to draw from, not {freed}"
    );

    create_reuse_log(&session, "log");
    let batch: Vec<i32> = (0..CHUNK as i32).collect();
    {
        let mut ds = session.dataset("log").unwrap();
        for _ in 0..ROUNDS {
            ds.append(&batch).unwrap();
        }
    }
    let spent = freed - session.space_accounting().unwrap().reusable_free_bytes;

    // A staged commit after the appends: it allocates metadata (an object header,
    // the rewritten managers) and raw data of its own, so it is the write that
    // would take up a page-alignment remainder the draws left misfiled.
    session
        .root()
        .create_dataset("after", |b| {
            b.with_i32_data(&vec![5i32; 4096]);
        })
        .unwrap();
    session.commit().unwrap();
    session.close().unwrap();

    // The run has to have *spent* the reserve, or a session that quietly appended
    // at end-of-file would satisfy the homogeneity check trivially.
    assert!(
        spent >= ROUNDS as u64 * CHUNK * 4 / 2,
        "the appends should have come out of the hole, but reusable free space \
         fell by only {spent}"
    );

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("log").unwrap().read_i32().unwrap().len(),
        ROUNDS * CHUNK as usize
    );
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    assert_eq!(
        file.dataset("after").unwrap().read_i32().unwrap(),
        vec![5i32; 4096]
    );
    drop(file);
    assert_no_persisted_free_space_holds_live_chunks(&path, &["log"]);
    assert_pages_homogeneous(&path, RECLAIM_PAGE, &["log", "keep", "ceiling", "after"]);
}

/// The page size the paged reclaim tests below create their files with. Large
/// enough that a page holds several objects, which is what makes the reclaim
/// question interesting rather than trivially per-object.
const RECLAIM_PAGE: u64 = 16384;
/// Rows appended onto each dataset of a populated churn cycle: a few kilobytes,
/// far under the megabyte an immediate append draws at most.
const CHURN_ROWS: usize = 1024;

/// Open `path` for editing without an `fsync` per write: every loop below closes
/// its session, and `SyncPolicy::OnClose` writes byte-identical files.
fn open_rw_on_close(path: &std::path::Path) -> File {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap()
}

/// Create a paged (or flat) persisting file holding one long-lived group, and
/// return the session editing it.
fn churn_fixture(path: &std::path::Path, strategy: FileSpaceStrategy) -> File {
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(strategy, true, 0);
    if strategy == FileSpaceStrategy::Page {
        b.with_file_space_page_size(RECLAIM_PAGE);
    }
    // Metadata only: a group carrying an attribute. It keeps the space each cycle
    // vacates interior rather than trailing without putting raw data in the pages
    // the scratch datasets' chunk indexes go into, which is what a real workload's
    // long-lived *metadata* looks like to the allocator.
    let mut keep = b.create_group("keep");
    keep.set_attr("kind", AttrValue::I64(1));
    b.add_group(keep.finish());
    b.write(path).unwrap();
    open_rw_on_close(path)
}

/// Stage a scratch group holding one empty chunked dataset, `resizable` or of
/// fixed shape, and return how much free space deleting it gives back.
fn scratch_delete_releases(
    strategy: FileSpaceStrategy,
    resizable: bool,
    name: &str,
) -> (File, u64) {
    let path = temp_path(name);
    let f = churn_fixture(&path, strategy);
    f.root()
        .create_group_with("scratch", |g| {
            g.create_dataset("log", |b| {
                b.with_i32_data(&[]).with_shape(&[0]).with_chunks(&[64]);
                if resizable {
                    b.with_maxshape(&[u64::MAX]);
                }
            });
        })
        .unwrap();
    f.commit().unwrap();
    let before = f.space_accounting().unwrap().reusable_free_bytes;
    f.root().delete("scratch").unwrap();
    f.commit().unwrap();
    let after = f.space_accounting().unwrap().reusable_free_bytes;
    (f, after - before)
}

/// How many whole, page-aligned free pages the session is offering back.
fn whole_free_pages(f: &File) -> usize {
    f.space_accounting()
        .unwrap()
        .reusable_free_space
        .iter()
        .filter(|&&(addr, len)| addr % RECLAIM_PAGE == 0 && len >= RECLAIM_PAGE)
        .count()
}

/// Deleting an empty *resizable* chunked dataset on a paged file returns its
/// chunk index, not only its object header (issue #388).
///
/// Such a dataset has no chunk data at all and still carries an index: this crate
/// builds an Extensible Array eagerly for every empty resizable dataset, because
/// an in-place append needs the index to exist before the first chunk arrives.
/// The paged reclaim proved an index raw only where chunk data abutted it, so
/// this shape — the one shape with no chunk data to abut — was dropped on every
/// delete, and the raw page it sat in could never be shown to be empty.
///
/// The index's size is not written down anywhere, so the flat strategy measures
/// it: the extra space its delete gives back when the deleted dataset is
/// resizable (an index) rather than fixed-shape (no index at all, just the
/// undefined address). That difference cancels the two object headers and the
/// commit tail, which are the same either way.
#[test]
fn paged_delete_of_an_empty_extensible_dataset_reclaims_its_index() {
    // The control: a flat persisting file has no page types to keep apart, so it
    // reclaims every index and this difference is the index itself.
    let (flat, with_index) = scratch_delete_releases(
        FileSpaceStrategy::FsmAggr,
        true,
        "hdf5_pure_fs_ea_flat_ext.h5",
    );
    flat.close().unwrap();
    let (flat, without_index) = scratch_delete_releases(
        FileSpaceStrategy::FsmAggr,
        false,
        "hdf5_pure_fs_ea_flat_fixed.h5",
    );
    flat.close().unwrap();
    let index_bytes = with_index - without_index;
    assert!(
        index_bytes >= 256,
        "the control must measure a real Extensible Array, not {index_bytes} bytes"
    );

    let path = temp_path("hdf5_pure_fs_ea_paged.h5");
    let f = churn_fixture(&path, FileSpaceStrategy::Page);
    f.root()
        .create_group_with("scratch", |g| {
            g.create_dataset("log", |b| {
                b.with_i32_data(&[])
                    .with_shape(&[0])
                    .with_chunks(&[64])
                    .with_maxshape(&[u64::MAX]);
            });
        })
        .unwrap();
    f.commit().unwrap();
    // The index is the only thing in its raw page, so while it is live that page
    // is not on offer — which is what makes the assertion after the delete about
    // the index and not about a page that was already free.
    assert_eq!(
        whole_free_pages(&f),
        0,
        "the live index must hold its page back"
    );
    let before = f.space_accounting().unwrap().reusable_free_bytes;
    f.root().delete("scratch").unwrap();
    f.commit().unwrap();
    let released = f.space_accounting().unwrap().reusable_free_bytes - before;
    assert!(
        released >= index_bytes,
        "a paged delete must give back the chunk index too ({released} bytes for \
         a subtree whose index alone is {index_bytes})"
    );
    assert!(
        whole_free_pages(&f) >= 1,
        "the raw page the index had to itself must come back whole"
    );
    f.close().unwrap();
    assert_eof_matches_file(&path);
}

/// The reporter's loop (issue #388): a paged persisting file that creates a
/// scratch group of empty resizable datasets, commits, deletes it, and commits
/// again reaches a steady size instead of growing a kilobyte per cycle.
///
/// A long-lived `keep` group goes in first so the space each cycle vacates is
/// interior rather than trailing, which is what makes reuse — not truncation —
/// the only way the file can stop growing.
#[test]
fn paged_group_churn_with_empty_datasets_reaches_a_steady_size() {
    const CYCLES: usize = 40;

    let path = temp_path("hdf5_pure_fs_paged_empty_churn.h5");
    let f = churn_fixture(&path, FileSpaceStrategy::Page);
    let mut sizes = Vec::with_capacity(CYCLES);
    let mut used = Vec::with_capacity(CYCLES);
    for cycle in 0..CYCLES {
        let name = format!("scratch_{cycle}");
        f.root()
            .create_group_with(&name, |g| {
                for i in 0..3 {
                    g.create_dataset(&format!("log{i}"), |b| {
                        b.with_i32_data(&[])
                            .with_shape(&[0])
                            .with_chunks(&[64])
                            .with_maxshape(&[u64::MAX]);
                    });
                }
            })
            .unwrap();
        f.commit().unwrap();
        f.root().delete(&name).unwrap();
        f.commit().unwrap();
        let acct = f.space_accounting().unwrap();
        sizes.push(f.file_size());
        used.push(f.file_size() - acct.reusable_free_bytes);
    }
    f.close().unwrap();
    assert_churn_settled(&sizes, &used);
    assert_churn_survivors(&path);
}

/// The reporter's loop for issue #413: a persisting file that creates a group of
/// resizable datasets, appends a payload onto each **in place**, commits, deletes
/// the group, commits again, and repeats, reaches a steady size.
///
/// Returns the file size and the space neither live nor reusable after every
/// cycle, and leaves one populated group in the closed file for the caller to
/// inspect.
///
/// The payload is a few kilobytes per dataset, far under the megabyte an
/// immediate append draws from the on-disk managers at most. A *floor* of that
/// size on the draw meant no hole this loop left was ever spent, so the file
/// grew by a payload (a page, on a paged file) per cycle while reporting all of
/// it as reusable — the empty-dataset loop of issue #388 and the staged append
/// were both stable, and this in-place one was not. A long-lived `keep` group
/// goes in first so the space each cycle vacates is interior rather than
/// trailing, which makes reuse, not truncation, the only way to stop growing.
fn populated_group_churn(
    path: &std::path::Path,
    strategy: FileSpaceStrategy,
    cycles: usize,
) -> (Vec<u64>, Vec<u64>) {
    let f = churn_fixture(path, strategy);
    let mut sizes = Vec::with_capacity(cycles);
    let mut used = Vec::with_capacity(cycles);
    let payload: Vec<i32> = (0..CHURN_ROWS as i32).collect();
    let populate = |name: &str| {
        f.root()
            .create_group_with(name, |g| {
                for i in 0..3 {
                    g.create_dataset(&format!("log{i}"), |b| {
                        b.with_i32_data(&[])
                            .with_shape(&[0])
                            .with_chunks(&[64])
                            .with_maxshape(&[u64::MAX]);
                    });
                }
            })
            .unwrap();
        f.commit().unwrap();
        for i in 0..3 {
            f.dataset(&format!("{name}/log{i}"))
                .unwrap()
                .append(&payload)
                .unwrap();
        }
        f.commit().unwrap();
    };
    for cycle in 0..cycles {
        let name = format!("scratch_{cycle}");
        populate(&name);
        f.root().delete(&name).unwrap();
        f.commit().unwrap();
        let acct = f.space_accounting().unwrap();
        sizes.push(f.file_size());
        used.push(f.file_size() - acct.reusable_free_bytes);
    }
    populate("last");
    f.close().unwrap();
    (sizes, used)
}

/// A churn loop's steady state: the last third of the run adds nothing, to the
/// file or to the space that is neither live nor reusable.
///
/// Compared as the *peak* over each of the last two thirds rather than as one
/// size, because a steady state is a short cycle rather than a fixed point once a
/// commit releases the free run at the end of the file (issue #418): the cycle
/// whose delete leaves the tail free gives those bytes back, and the next cycle
/// takes them again. What may not move is the ceiling of that cycle.
fn assert_churn_settled(sizes: &[u64], used: &[u64]) {
    let third = sizes.len() / 3;
    let peak = |xs: &[u64]| xs.iter().copied().max().expect("a non-empty run");
    assert_eq!(
        peak(&sizes[2 * third..]),
        peak(&sizes[third..2 * third]),
        "the file must stop growing once the churn is in its steady state \
         (sizes: {sizes:?})"
    );
    assert_eq!(
        peak(&used[2 * third..]),
        peak(&used[third..2 * third]),
        "space that is neither live nor reusable must stop accumulating \
         (used: {used:?})"
    );
}

/// What every churn loop over a [`churn_fixture`] leaves behind: the long-lived
/// group intact, every scratch group gone, and the recorded end-of-file true.
fn assert_churn_survivors(path: &std::path::Path) {
    let f = File::open(path).unwrap();
    assert_eq!(
        f.group("keep").unwrap().attrs().unwrap().get("kind"),
        Some(&AttrValue::I64(1))
    );
    assert!(f.group("scratch_0").is_err(), "every scratch group is gone");
    drop(f);
    assert_eof_matches_file(path);
}

/// The populated group [`populated_group_churn`] leaves behind reads back in
/// full.
fn assert_last_group_populated(path: &std::path::Path) {
    let f = File::open(path).unwrap();
    let want: Vec<i32> = (0..CHURN_ROWS as i32).collect();
    for i in 0..3 {
        assert_eq!(
            f.dataset(&format!("last/log{i}"))
                .unwrap()
                .read_i32()
                .unwrap(),
            want
        );
    }
}

/// A paged persisting file under the populated delete-and-recreate loop of
/// issue #413 reaches a steady size, and every page it reused holds one kind of
/// byte.
#[test]
fn paged_group_churn_with_populated_datasets_reaches_a_steady_size() {
    let path = temp_path("hdf5_pure_fs_paged_populated_churn.h5");
    let (sizes, used) = populated_group_churn(&path, FileSpaceStrategy::Page, 30);
    assert_churn_settled(&sizes, &used);
    assert_churn_survivors(&path);
    assert_last_group_populated(&path);
    let last = ["last/log0", "last/log1", "last/log2"];
    assert_pages_homogeneous(&path, RECLAIM_PAGE, &last);
    assert_no_persisted_free_space_holds_live_chunks(&path, &last);
}

/// The flat persisting strategy under the same loop (issue #413). The reporter
/// measured it as the one that recycled; it had the same floor on its draw, and
/// on top of that the draw's own manager rewrite had nowhere to put its tail once
/// the draw had emptied the file's one free list, so it appended a tail at
/// end-of-file per draw.
#[test]
fn persisting_group_churn_with_populated_datasets_reaches_a_steady_size() {
    let path = temp_path("hdf5_pure_fs_flat_populated_churn.h5");
    let (sizes, used) = populated_group_churn(&path, FileSpaceStrategy::FsmAggr, 30);
    assert_churn_settled(&sizes, &used);
    assert_churn_survivors(&path);
    assert_last_group_populated(&path);
    assert_no_persisted_free_space_holds_live_chunks(
        &path,
        &["last/log0", "last/log1", "last/log2"],
    );
}

/// Repeated `Dataset::append_staged` + `commit` on a paged file does not strand
/// the index each flush supersedes (issue #388).
///
/// Each flush rebuilds the dataset's Extensible Array and frees the old one. The
/// appended chunks and the rebuilt index go down as one blob, which is what lets
/// the *next* flush prove that the index it supersedes sits in a raw page;
/// placing them separately let reuse drop a chunk into the hole a previous index
/// left and strand the new index against nothing, and the file then accumulated
/// one dead index per flush.
#[test]
fn paged_staged_append_churn_does_not_leak_the_old_index() {
    const FLUSHES: usize = 200;
    const ROWS: usize = 16;

    let path = temp_path("hdf5_pure_fs_paged_append_churn.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(RECLAIM_PAGE);
    b.create_dataset("d")
        .with_i32_data(&[0i32; ROWS])
        .with_shape(&[ROWS as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[ROWS as u64]);
    b.write(&path).unwrap();

    let f = open_rw_on_close(&path);
    for flush in 0..FLUSHES {
        f.dataset("d")
            .unwrap()
            .append_staged(|b| {
                b.append_i32(&[flush as i32; ROWS]);
            })
            .unwrap();
        f.commit().unwrap();
    }
    let acct = f.space_accounting().unwrap();
    let used = f.file_size() - acct.reusable_free_bytes;
    // What the file legitimately holds: the live rows, plus the one live index
    // over them and the headers around it, which two pages cover comfortably.
    let payload = ((FLUSHES + 1) * ROWS * 4) as u64;
    let ceiling = payload + 2 * RECLAIM_PAGE;
    assert!(
        used <= ceiling,
        "{FLUSHES} staged appends left {used} bytes neither live nor reusable, \
         over the {ceiling} that {payload} bytes of rows plus two pages allow"
    );

    f.close().unwrap();
    let f = File::open(&path).unwrap();
    let want: Vec<i32> = std::iter::repeat_n(0, ROWS)
        .chain((0..FLUSHES).flat_map(|n| std::iter::repeat_n(n as i32, ROWS)))
        .collect();
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), want);
    drop(f);
    assert_eof_matches_file(&path);
    // Reclaiming the superseded index is only worth doing if it is reclaimed
    // under the page type it actually sits in: filing a metadata region as raw
    // mixes a page (issue #261), and offering a live chunk as free hands the next
    // session the bytes the dataset is stored in (issue #387). Neither shows up
    // in the size ceiling above, which a wrong answer satisfies best of all.
    assert_pages_homogeneous(&path, RECLAIM_PAGE, &["d"]);
    assert_no_persisted_free_space_holds_live_chunks(&path, &["d"]);
}

/// A staged append that adds several chunks fills the chunk-sized holes an
/// earlier commit left, rather than appending the lot at end-of-file.
///
/// The appended chunks and the rebuilt index are laid down as one blob where a
/// single freed region holds them, since that contiguity is what lets the *next*
/// commit place the index it supersedes on a paged file. It must not be bought
/// with reuse: a run of chunks placed one at a time fills several small holes
/// that no single-hole reservation can reach, and requiring one region for the
/// whole blob sent every such append past the end of the file instead.
///
/// The fixture leaves four holes just wider than one chunk, kept apart by live
/// spacers so they cannot coalesce into a region the blob would fit, and appends
/// four chunks into them.
#[test]
fn a_multi_chunk_staged_append_fills_chunk_sized_holes() {
    const CHUNK: usize = 64; // elements, so 256 bytes of i32 per chunk
    const APPENDED: usize = 4;
    let payload = (APPENDED * CHUNK * 4) as u64;

    for (strategy, page, name) in [
        (
            FileSpaceStrategy::FsmAggr,
            0,
            "hdf5_pure_fs_append_holes_flat.h5",
        ),
        (
            FileSpaceStrategy::Page,
            4096,
            "hdf5_pure_fs_append_holes_paged.h5",
        ),
    ] {
        let path = temp_path(name);
        let mut b = FileBuilder::new();
        b.with_file_space_strategy(strategy, true, 0);
        if page > 0 {
            b.with_file_space_page_size(page);
        }
        b.create_dataset("d")
            .with_i32_data(&(0..CHUNK as i32).collect::<Vec<i32>>())
            .with_shape(&[CHUNK as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[CHUNK as u64]);
        // Scratch and spacer alternate, so deleting the scratch datasets leaves
        // four separated holes rather than one run.
        for i in 0..APPENDED {
            for role in ["scratch", "spacer"] {
                b.create_dataset(&format!("{role}{i}"))
                    .with_i32_data(&(0..72).collect::<Vec<i32>>());
            }
        }
        b.write(&path).unwrap();

        let f = open_rw_on_close(&path);
        for i in 0..APPENDED {
            f.root().delete(&format!("scratch{i}")).unwrap();
        }
        f.commit().unwrap();
        let (size_before, free_before) = (
            f.file_size(),
            f.space_accounting().unwrap().reusable_free_bytes,
        );
        assert!(
            free_before >= payload,
            "{name}: the fixture must free at least the bytes the append will place"
        );

        f.dataset("d")
            .unwrap()
            .append_staged(|b| {
                b.append_i32(&(0..(APPENDED * CHUNK) as i32).collect::<Vec<i32>>());
            })
            .unwrap();
        f.commit().unwrap();
        let (size_after, free_after) = (
            f.file_size(),
            f.space_accounting().unwrap().reusable_free_bytes,
        );
        assert!(
            size_after - size_before < payload,
            "{name}: an append of {payload} bytes of chunks into holes that fit them \
             must not grow the file by that much (grew {})",
            size_after - size_before
        );
        assert!(
            free_after < free_before,
            "{name}: the holes must be spent, not left untouched \
             ({free_before} before, {free_after} after)"
        );

        f.close().unwrap();
        // The paged half places the appended chunks and the rebuilt index into
        // holes; both must land in raw pages, and neither may leave a live chunk
        // inside a region the managers advertise.
        if page > 0 {
            assert_pages_homogeneous(&path, page, &["d"]);
        }
        assert_no_persisted_free_space_holds_live_chunks(&path, &["d"]);
        let f = File::open(&path).unwrap();
        assert_eq!(
            f.dataset("d").unwrap().read_i32().unwrap(),
            (0..CHUNK as i32)
                .chain(0..(APPENDED * CHUNK) as i32)
                .collect::<Vec<i32>>()
        );
        for i in 0..APPENDED {
            assert_eq!(
                f.dataset(&format!("spacer{i}"))
                    .unwrap()
                    .read_i32()
                    .unwrap(),
                (0..72).collect::<Vec<i32>>()
            );
        }
        drop(f);
        assert_eof_matches_file(&path);
    }
}
