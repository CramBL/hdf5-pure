// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets; skip elsewhere so the pure-Rust suite still
// runs under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Interop for the owned SWMR writer (issue #148, PR B): after a clean
//! `File::close`, the SWMR-write flag is cleared and the reference C library
//! reads the streamed appends back exactly.

use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

fn build_swmr(path: &std::path::Path, n: i32, chunk: u64) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    b.write(path).unwrap();
}

#[test]
fn c_library_reads_swmr_appends_after_close() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("s.h5");
    build_swmr(&path, 50, 1); // chunk length 1: the common streaming layout

    {
        let file = File::open_swmr_writer(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        // Several chunk-aligned appends across the inline -> direct -> super-block
        // boundaries of the Extensible-Array index.
        ds.append(&(50..120).collect::<Vec<i32>>()).unwrap();
        ds.append(&(120..250).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let f = hdf5::File::open(&path).unwrap();
    let v = f.dataset("d").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(v, (0..250).collect::<Vec<_>>());
    f.close().unwrap();
}

/// The status-flags field of a *version-1* superblock is read from the offset
/// the C library writes it to. The v1 layout puts the flags before the chunk
/// B-tree K, and this crate had the two swapped, so it reported a file's
/// `istore_k` as its status flags. Only a file the C library wrote can settle
/// the order — a hand-built fixture asserting the crate's own layout agrees
/// with itself whichever way round it is.
///
/// The flag check skips a v1 superblock (see `file_lock::check_status_flags`),
/// so this is about `File::superblock()` reporting the truth, not about the
/// refusal — but it is the same byte, and a gate that ever widened would have
/// refused every v1 file whose `istore_k` happens to be odd.
#[test]
fn a_v1_superblock_reports_the_flags_the_c_library_wrote() {
    use hdf5::file::LibraryVersion;

    let dir = tempdir().unwrap();
    let path = dir.path().join("legacy.h5");
    // A non-default, odd chunk B-tree K: it sits beside the flags in the v1
    // layout, and reading it as the flags would report bit 0 — "open for write".
    const ISTORE_K: u32 = 33;
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::Earliest, LibraryVersion::V18))
            .with_fcpl(|p| p.istore_k(ISTORE_K))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((4,))
            .chunk((2,))
            .create("d")
            .unwrap()
            .write(&[0i32, 1, 2, 3])
            .unwrap();
        assert_eq!(file.create_plist().unwrap().istore_k(), ISTORE_K);
        file.close().unwrap();
    }

    let sb = File::open(&path).unwrap();
    let sb = sb.superblock();
    assert_eq!(sb.version, 1, "the earliest format with istore_k gives v1");
    assert_eq!(
        sb.indexed_storage_internal_node_k,
        Some(ISTORE_K as u16),
        "the chunk B-tree K must read back as itself"
    );
    assert_eq!(
        sb.consistency_flags, 0,
        "a cleanly closed file has no status flags set"
    );
}

/// Both libraries read the same superblock status-flags byte the same way
/// (issue #245). On a file a crashed SWMR writer left flagged, a plain open is
/// refused by each and a SWMR read is accepted by each; after `clear_swmr_flag`
/// — the `h5clear -s` equivalent — every open works again on both sides.
///
/// This is the divergence the issue reported: hdf5-pure used to open, and edit
/// in place, a file `H5Fopen` refuses.
#[test]
fn both_libraries_refuse_a_file_a_crashed_swmr_writer_left_flagged() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("flagged.h5");
    build_swmr(&path, 4, 4);

    // A crashed writer: leak the handle so neither `close` nor `Drop` clears the
    // flag it raised.
    #[expect(clippy::mem_forget, reason = "the test models a writer that crashed")]
    std::mem::forget(File::open_swmr_writer(&path).unwrap());

    let c_err = hdf5::File::open(&path).expect_err("the C library refuses a flagged file");
    assert!(
        c_err.to_string().contains("already open for write"),
        "the C library refused it for some other reason: {c_err}"
    );
    assert!(
        matches!(File::open(&path), Err(hdf5_pure::Error::FileMarkedInUse(_))),
        "this crate must refuse what the C library refuses"
    );

    // A SWMR read is what the flag exists to permit, in both libraries.
    hdf5::File::open_as(&path, hdf5::OpenMode::ReadSWMR)
        .expect("the C library attaches a SWMR reader to a flagged file")
        .close()
        .unwrap();
    File::open_swmr(&path).expect("this crate attaches a SWMR reader too");

    File::clear_swmr_flag(&path).unwrap();
    hdf5::File::open(&path)
        .expect("clearing the flag recovers the file for the C library")
        .close()
        .unwrap();
    File::open(&path).expect("and for this crate");
}
