//! File-space strategy and persistent free-space *reuse*.
//!
//! Mirroring `H5Pset_file_space_strategy` / `H5Pset_file_space_page_size`, a
//! written file can record how it manages free space. With `persist = true`, the
//! regions a `File::open_rw` session frees are written to on-disk free-space-manager
//! blocks, so a *later* session (this crate's or the reference C library's) seeds
//! its free list from them and writes new objects into the holes instead of
//! growing the file.
//!
//! It also gives space back. When the region a delete frees runs to the end of
//! the file, the commit shortens the file to just above its last live allocation
//! rather than carrying the freed bytes as a hole, so `File::file_size()` is not
//! a high-water mark.
//!
//! This example shows both halves on one add/delete/re-add cycle: the file
//! shrinks on the delete, and the re-add takes the space back without the file
//! ever exceeding what it measured before. The same churn runs on a *paged* file
//! (`FileSpaceStrategy::Page`), which works in whole pages, and on a
//! non-persisting file as a control, where the freed space is forgotten on close
//! — that file keeps both copies and ends up twice the size.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example file_space
//! ```

use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};
use std::path::Path;

/// 4096 f64 values = 32 KiB of raw data — large enough that whether the file
/// takes the space back or keeps a second copy is unmistakable in the byte counts.
const SCRATCH_LEN: usize = 4096;
const SCRATCH_BYTES: u64 = (SCRATCH_LEN * size_of::<f64>()) as u64;

/// What a re-add may cost over the size the file had before the delete: the
/// free-space-manager blocks the persisting file now carries, plus a page of
/// alignment on the paged one. Not another copy of the data, which is the whole
/// point — a quarter of it is a line neither the manager blocks nor a page comes
/// anywhere near, and a second copy clears by four times over.
const OVERHEAD_ALLOWANCE: u64 = SCRATCH_BYTES / 4;

/// The file's length at the three points of one add/delete/re-add cycle.
struct Cycle {
    /// After `scratch` was added: the high-water mark the file reaches.
    before_delete: u64,
    /// After it was deleted in its own session.
    after_delete: u64,
    /// After a same-sized `scratch2` was added in a third session.
    after_readd: u64,
}

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");

    // ---- Persisting file: free space is reused across sessions ----------
    let persisting = dir.path().join("managed.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    // A free-space-manager strategy that persists across sessions. Its numbers are
    // the data's own; the paged run below reports the same reuse in whole pages.
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1); // strategy, persist, threshold
    builder.write(&persisting).expect("write file");

    // The strategy is stored in a superblock-extension message and survives a
    // reopen (the reference C library observes it too).
    let strategy = File::open(&persisting).unwrap().file_space_strategy();
    assert_eq!(strategy, Some(FileSpaceStrategy::FsmAggr));
    println!("recorded strategy: {strategy:?}\n");

    println!("persisting file (strategy = FsmAggr, persist = true):");
    let flat = churn(&persisting);
    assert_released_and_reclaimed("FsmAggr", &flat);

    // ---- Paged file: the same cycle, in whole pages ---------------------
    // A paged file segregates metadata and raw data by page, so it tracks free
    // space per page type and an allocation draws from the list matching what it
    // is placing (or from a page that is wholly free). The end of allocation stays
    // a whole number of pages, so what a delete gives back here is whole pages.
    let paged = dir.path().join("paged.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    builder.write(&paged).expect("write file");

    println!("\npaged file (strategy = Page, persist = true):");
    let paged_cycle = churn(&paged);
    assert_released_and_reclaimed("Page", &paged_cycle);

    // ---- Control: a non-persisting file forgets its free list -----------
    // Identical churn on a default-strategy file. Here the free list lives only
    // for the open session and is discarded on close, so the session that deletes
    // `scratch` cannot record the space it freed and the session that adds
    // `scratch2` has no record to draw on: the file keeps both copies.
    let default = dir.path().join("default.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.write(&default).expect("write file");

    println!("\ncontrol — non-persisting file (default strategy):");
    let control = churn(&default);

    let control_growth = control.after_readd - control.after_delete;
    assert!(
        control_growth >= SCRATCH_BYTES,
        "expected no cross-session reuse: re-adding {SCRATCH_BYTES} bytes should \
         grow the file by at least that, got {control_growth}"
    );
    assert!(
        control.after_readd > control.before_delete + SCRATCH_BYTES / 2,
        "expected a high-water mark: the control ended at {} after peaking at {} \
         before the delete, so it did not keep a second copy",
        control.after_readd,
        control.before_delete
    );

    // ---- Every file still holds the original and the re-added data ------
    for path in [&persisting, &paged, &default] {
        let file = File::open(path).unwrap();
        assert_eq!(
            file.dataset("keep").unwrap().read_i32().unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            file.dataset("scratch2").unwrap().read_f64().unwrap(),
            vec![7.0; SCRATCH_LEN]
        );
    }

    println!(
        "\nverified: a delete released {} bytes (FsmAggr) and {} (Page), and \
         re-adding {SCRATCH_BYTES} bytes left those files at {} and {} against \
         peaks of {} and {} — while the non-persisting control ended at {} \
         against a peak of {}",
        flat.before_delete - flat.after_delete,
        paged_cycle.before_delete - paged_cycle.after_delete,
        flat.after_readd,
        paged_cycle.after_readd,
        flat.before_delete,
        paged_cycle.before_delete,
        control.after_readd,
        control.before_delete
    );
}

/// The two halves of the persisting story, on one cycle: the delete gave the end
/// of the file back, and taking the same space again did not push the file past
/// the mark it had already reached.
fn assert_released_and_reclaimed(label: &str, c: &Cycle) {
    assert!(
        c.after_delete + SCRATCH_BYTES / 2 < c.before_delete,
        "{label}: deleting a {SCRATCH_BYTES}-byte dataset that ran to the end of \
         the file should have shortened it well past half that ({} bytes, from {})",
        c.after_delete,
        c.before_delete
    );
    assert!(
        c.after_readd <= c.before_delete + OVERHEAD_ALLOWANCE,
        "{label}: re-adding {SCRATCH_BYTES} bytes left the file at {}, past the \
         {} it had reached before the delete plus the {OVERHEAD_ALLOWANCE} bytes \
         of manager blocks and page alignment a re-add may cost",
        c.after_readd,
        c.before_delete
    );
}

/// Add a dataset, delete it, then add a same-sized one in a *fresh* session,
/// reporting the file's length at each step. Whether the delete shortened the
/// file and whether the re-add pushed it past its old peak are what the three
/// numbers reveal.
fn churn(path: &Path) -> Cycle {
    // Add `scratch`, then delete it in its own session so the region is freed.
    let session = File::open_rw(path).expect("open for editing");
    session
        .root()
        .create_dataset("scratch", |b| {
            b.with_f64_data(&vec![0.0; SCRATCH_LEN]);
        })
        .unwrap();
    session.commit().unwrap();
    drop(session); // release the editor's exclusive lock before the next session
    let before_delete = len(path);
    println!("  after adding {SCRATCH_BYTES} bytes: {before_delete} bytes");

    let session = File::open_rw(path).expect("reopen for editing");
    session.root().delete("scratch").unwrap();
    session.commit().unwrap();
    drop(session); // release the lock before reading the file back
    let after_delete = len(path);

    // For a persisting file, whatever the delete did not give back to the
    // filesystem is recorded on disk and a later session draws on it; for a
    // non-persisting file the list is empty here.
    let free = File::open(path).unwrap().persisted_free_space().unwrap();
    let total_free: u64 = free.iter().map(|&(_, len)| len).sum();
    println!(
        "  after delete: {after_delete} bytes, {} persisted free region(s) ({total_free} bytes)",
        free.len()
    );

    // Add a same-sized dataset in a fresh session and measure the growth.
    let session = File::open_rw(path).expect("reopen for editing");
    session
        .root()
        .create_dataset("scratch2", |b| {
            b.with_f64_data(&vec![7.0; SCRATCH_LEN]);
        })
        .unwrap();
    session.commit().unwrap();
    drop(session); // release the lock before reading the file back
    let after_readd = len(path);
    let growth = after_readd - after_delete;
    println!(
        "  after re-adding {SCRATCH_BYTES} bytes of new data: {after_readd} bytes (+{growth})"
    );
    Cycle {
        before_delete,
        after_delete,
        after_readd,
    }
}

fn len(path: &Path) -> u64 {
    std::fs::metadata(path).unwrap().len()
}
