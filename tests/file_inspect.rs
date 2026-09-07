//! Tests for the file-inspection helpers added for issue #32 (Group A + B):
//! `is_hdf5` / `is_hdf5_bytes`, `File::file_size`, `File::libver_bound`, and
//! `FileBuilder::with_libver_bounds`.

use hdf5_pure::{File, FileBuilder, LibVer, is_hdf5, is_hdf5_bytes};

use temp::temp_path;
use test_util::temp;

fn sample_file() -> Vec<u8> {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0]);
    builder.finish().unwrap()
}

#[test]
fn is_hdf5_bytes_detects_signature() {
    let bytes = sample_file();
    assert!(is_hdf5_bytes(&bytes));
    assert!(!is_hdf5_bytes(&[0u8; 64]));
    assert!(!is_hdf5_bytes(&[]));
}

#[test]
fn is_hdf5_path_roundtrip() {
    // One directory, three names: this test wants a missing path alongside the
    // two it writes, so it holds the directory rather than a single fixture.
    let held = tempfile::tempdir().unwrap();
    let dir = held.path();
    let h5 = dir.join("hdf5_pure_is_hdf5_yes.h5");
    let other = dir.join("hdf5_pure_is_hdf5_no.bin");
    std::fs::write(&h5, sample_file()).unwrap();
    std::fs::write(&other, b"not an hdf5 file at all").unwrap();

    assert!(is_hdf5(&h5).unwrap());
    assert!(!is_hdf5(&other).unwrap());

    // A missing file is an I/O error, not `Ok(false)`.
    assert!(is_hdf5(dir.join("hdf5_pure_definitely_missing.h5")).is_err());
}

#[test]
fn file_size_matches_buffer_and_metadata() {
    let bytes = sample_file();
    let len = bytes.len() as u64;

    let file = File::from_bytes(bytes.clone()).unwrap();
    assert_eq!(file.file_size(), len);

    let path = temp_path("hdf5_pure_file_size.h5");
    std::fs::write(&path, &bytes).unwrap();
    let on_disk = File::open(&path).unwrap();
    assert_eq!(on_disk.file_size(), len);
    assert_eq!(on_disk.file_size(), std::fs::metadata(&path).unwrap().len());
}

#[test]
fn libver_bound_reports_writer_format() {
    // Unbounded, this crate writes a version 3 superblock — the HDF5 1.10 format.
    let file = File::from_bytes(sample_file()).unwrap();
    assert_eq!(file.superblock().version, 3);
    assert_eq!(file.libver_bound(), LibVer::V110);
    assert_eq!(file.libver_bound(), LibVer::WRITER_DEFAULT);
}

#[test]
fn libver_bounds_accept_straddling_range() {
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::LATEST);
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    // Earliest..=Latest straddles the produced 1.10 format → accepted.
    assert!(builder.finish().is_ok());
}

#[test]
fn libver_bounds_reject_too_old_upper_bound() {
    let mut builder = FileBuilder::new();
    // An upper bound of `Earliest` asks for a version 0 or 1 superblock, which
    // pairs with the v1 symbol-table groups this crate reads but does not write.
    builder.with_libver_bounds(LibVer::Earliest, LibVer::Earliest);
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    let err = builder.finish().unwrap_err();
    assert!(
        err.to_string().contains("library-version bounds"),
        "unexpected error: {err}"
    );
}

/// An upper bound of 1.8 is satisfiable — it selects the older format rather
/// than failing, which is what a file destined for MATLAB's `load` needs.
#[test]
fn libver_bounds_of_1_8_select_the_older_format() {
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    let bytes = builder.finish().expect("1.8 is a format this crate writes");
    assert_eq!(bytes[8], 2, "version 2 superblock");

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.libver_bound(), LibVer::V18);
    assert_eq!(file.libver_bound(), LibVer::WRITER_OLDEST);
    assert_eq!(
        file.dataset("data").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

/// A lower bound newer than the 1.10 format this crate writes licenses newer
/// encodings without requiring them — the meaning it has in
/// `H5Pset_libver_bounds` — so it writes the 1.10 format rather than failing.
#[test]
fn libver_bounds_above_the_written_format_still_write_it() {
    for (low, high) in [
        (LibVer::V112, LibVer::LATEST),
        (LibVer::LATEST, LibVer::LATEST),
    ] {
        let mut builder = FileBuilder::new();
        builder.with_libver_bounds(low, high);
        builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
        let bytes = builder
            .finish()
            .expect("a lower bound above 1.10 is satisfied by the 1.10 format");
        assert_eq!(bytes[8], 3, "version 3 superblock");

        let file = File::from_bytes(bytes).unwrap();
        assert_eq!(file.libver_bound(), LibVer::V110);
        assert_eq!(file.libver_bound(), LibVer::WRITER_DEFAULT);
        assert_eq!(
            file.dataset("data").unwrap().read_i32().unwrap(),
            vec![1, 2, 3]
        );
    }
}
