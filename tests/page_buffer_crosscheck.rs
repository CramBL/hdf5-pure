// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets; skip elsewhere so the pure-Rust suite still
// runs under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! The write-back page buffer against the reference C library (issue #308).
//!
//! Two claims are made for the crash mark a page-buffered session raises, and
//! only the C library can settle either. The first is that a session which dies
//! with dirty pages in memory leaves a file the *whole ecosystem* refuses, not
//! just this crate — the mark is worth its cost only if `H5Fopen` honors it, and
//! `H5F_SUPER_WRITE_ACCESS` alone is a different bit pattern from the SWMR pair
//! whose refusal `owned_swmr_crosscheck` already pins. The second is that a file
//! written *through* the buffer and closed cleanly is an ordinary file again:
//! mark down, paged layout intact, data where the C library looks for it.

use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5_pure::{
    File, FileAccessProperties, FileBuilder, FileLocking, FileSpaceStrategy, SyncPolicy,
};
use tempfile::tempdir;

/// An appendable file, paged or not. A page buffer requires neither (issue
/// #357), which is why the reading crosscheck below runs on both.
fn build(path: &std::path::Path, paged: bool) {
    let mut b = FileBuilder::new();
    if paged {
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
            .with_file_space_page_size(4096);
    }
    b.create_dataset("d")
        .with_i32_data(&(0..64).collect::<Vec<i32>>())
        .with_shape(&[64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[64]);
    b.write(path).unwrap();
}

fn page_buffered() -> FileAccessProperties {
    FileAccessProperties::new()
        .with_sync_policy(SyncPolicy::OnClose)
        .with_locking(FileLocking::Disabled)
        .with_page_buffer_size(1 << 20)
}

/// The mark's whole justification: a page-buffered session that dies is refused
/// by the C library too, so its file cannot be read clean and believed anywhere.
///
/// The status flags here are `0x01` — the write bit alone — where a crashed SWMR
/// writer leaves `0x05`. That the C library refuses the pair says nothing about
/// whether it refuses this one, which is the bit this crate is newly raising.
#[test]
fn the_c_library_refuses_a_file_a_crashed_page_buffered_session_left_marked() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("crashed.h5");
    build(&path, true);

    // A crashed session: leak the handle so neither `close` nor `Drop` clears
    // the mark it raised.
    {
        let file = File::open_rw_with_options(&path, page_buffered()).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[7i32; 64]).unwrap();
        std::mem::forget(ds);
        std::mem::forget(file);
    }
    // Only what the C library can settle. That the byte is `0x01`, and that this
    // crate refuses it and recovers from it, are `superblock_status_flags`'
    // to assert — repeating them here costs a libhdf5 link to learn nothing.
    let c_err = hdf5::File::open(&path).expect_err("the C library refuses a marked file");
    assert!(
        c_err.to_string().contains("already open for write"),
        "the C library refused it for some other reason: {c_err}"
    );

    File::clear_swmr_flag(&path).unwrap();
    hdf5::File::open(&path)
        .expect("clearing the mark recovers the file for the C library")
        .close()
        .unwrap();
}

/// A session that closes cleanly leaves an ordinary file: the C library opens it
/// without complaint and reads back what was appended through the buffer — and
/// the file-space strategy it started with is the one it ends with.
///
/// The reads matter as much as the open. A page buffer merges by page, so a
/// merge that filled a gap between two runs with the wrong bytes would still
/// produce a file whose superblock and headers check out — the data is where
/// that would show.
///
/// Run on an **unpaged** file as well as a paged one. `H5Pset_page_buffer_size`
/// refuses the unpaged case outright (`H5PB_create`: "Enabling Page Buffering
/// requires PAGE file space strategy"), because the C page buffer is a page
/// cache whose per-kind reservations count pages the paged allocator segregates.
/// This crate's is a write gatherer that merges within a page-sized window, and
/// an unpaged file gets the format's default 4 KiB one — so the requirement was
/// inherited rather than needed (issue #357). The C library is what says the
/// result is a real HDF5 file either way: the crate reading back its own bytes
/// would not distinguish a correct merge from a consistently wrong one.
#[test]
fn the_c_library_reads_a_file_written_through_a_page_buffer() {
    let dir = tempdir().unwrap();
    for paged in [true, false] {
        let label = if paged { "paged" } else { "unpaged" };
        let path = dir.path().join(format!("buffered_{label}.h5"));
        build(&path, paged);

        {
            let file = File::open_rw_with_options(&path, page_buffered()).unwrap();
            for round in 0..4i32 {
                let mut ds = file.dataset("d").unwrap();
                ds.append(&vec![round; 64]).unwrap();
            }
            file.root()
                .create_dataset("added", |b| {
                    b.with_f64_data(&[2.5f64; 32]).with_shape(&[32]);
                })
                .unwrap();
            file.commit().unwrap();
            file.close().unwrap();
        }

        let f = hdf5::File::open(&path)
            .unwrap_or_else(|e| panic!("{label}: a cleanly closed buffered file opens: {e}"));
        let d = f.dataset("d").unwrap().read_raw::<i32>().unwrap();
        let mut expected: Vec<i32> = (0..64).collect();
        for round in 0..4i32 {
            expected.extend(std::iter::repeat_n(round, 64));
        }
        assert_eq!(
            d, expected,
            "{label}: the appends must read back through the C library"
        );
        assert_eq!(
            f.dataset("added").unwrap().read_raw::<f64>().unwrap(),
            vec![2.5f64; 32],
            "{label}: and so must the committed dataset"
        );
        let expected_strategy = if paged {
            CStrategy::FreeSpaceManager {
                paged: true,
                persist: true,
                threshold: 1,
            }
        } else {
            // The C library's own default: managers and aggregators, unpaged and
            // not persisting.
            CStrategy::FreeSpaceManager {
                paged: false,
                persist: false,
                threshold: 1,
            }
        };
        assert_eq!(
            f.create_plist().unwrap().file_space_strategy(),
            expected_strategy,
            "{label}: the file must still have the strategy it started with"
        );
        f.close().unwrap();
    }
}
