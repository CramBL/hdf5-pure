#![cfg(feature = "__hdf5-1.10")]
//! Native reads and file status flags for owned SWMR writers.

use std::time::Duration;

use hdf5::file::LibraryVersion;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::SuperblockStatus;
use hdf5_pure::SuperblockStatusFlags;
use hdf5_pure::SuperblockVersion;
use rstest::rstest;

use test_util::process::{self, Fixture, Readiness};
use test_util_hdf5::dataset::{self, Unlimited};

#[test]
fn c_library_reads_swmr_appends_after_close() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.h5");
    Unlimited::new("d", &(0..50).collect::<Vec<i32>>(), 1).pure_create(&path); // chunk length 1: the common streaming layout

    {
        let file = File::open_swmr_writer(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        // Several chunk-aligned appends across the inline -> direct -> super-block
        // boundaries of the Extensible-Array index.
        ds.append(&(50..120).collect::<Vec<i32>>()).unwrap();
        ds.append(&(120..250).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    assert_eq!(
        dataset::read_libhdf5::<i32>(&path, "d"),
        (0..250).collect::<Vec<_>>()
    );
}

#[rstest]
#[case(LibraryVersion::Earliest, 32, SuperblockVersion::Zero)]
#[case(LibraryVersion::Earliest, 33, SuperblockVersion::One)]
#[case(LibraryVersion::V18, 33, SuperblockVersion::Two)]
#[case(LibraryVersion::V110, 33, SuperblockVersion::Three)]
fn a_superblock_reports_the_flags_the_c_library_wrote(
    #[case] libver: LibraryVersion,
    #[case] indexed_storage_k: u32,
    #[case] version: SuperblockVersion,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(libver, LibraryVersion::V110))
            .with_fcpl(|p| p.istore_k(indexed_storage_k))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((4,))
            .chunk((2,))
            .create("d")
            .unwrap()
            .write(&[0i32, 1, 2, 3])
            .unwrap();
        assert_eq!(file.create_plist().unwrap().istore_k(), indexed_storage_k);
        file.close().unwrap();
    }

    let sb = File::open(&path).unwrap();
    let sb = sb.superblock();
    assert_eq!(sb.version(), version);
    if version == SuperblockVersion::One {
        assert_eq!(
            sb.indexed_storage_internal_node_k,
            Some(u16::try_from(indexed_storage_k).unwrap())
        );
    } else {
        assert_eq!(sb.indexed_storage_internal_node_k, None);
    }
    assert_eq!(sb.consistency_flags(), SuperblockStatusFlags::new());
    assert_eq!(
        sb.status(),
        match version {
            SuperblockVersion::Zero => SuperblockStatus::Zero(0),
            SuperblockVersion::One => SuperblockStatus::One(0),
            SuperblockVersion::Two => SuperblockStatus::Two(0),
            SuperblockVersion::Three => SuperblockStatus::Three(0),
        }
    );
}

#[test]
fn both_libraries_refuse_a_file_a_crashed_swmr_writer_left_flagged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("flagged.h5");
    Unlimited::new("d", &(0..4).collect::<Vec<i32>>(), 4).pure_create(&path);

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
        matches!(File::open(&path), Err(Error::FileMarkedInUse(_))),
        "this crate must refuse what the C library refuses"
    );

    // A SWMR read is what the flag exists to permit, in both libraries.
    hdf5::File::open_as(&path, hdf5::OpenMode::ReadSWMR)
        .expect("the C library attaches a SWMR reader to a flagged file")
        .close()
        .unwrap();
    File::open_swmr(&path).expect("this crate attaches a SWMR reader too");

    File::clear_status_flags(&path).unwrap();
    hdf5::File::open(&path)
        .expect("clearing the flag recovers the file for the C library")
        .close()
        .unwrap();
    File::open(&path).expect("and for this crate");
}

#[rstest]
#[case::ordinary_writer(false, SuperblockStatusFlags::write_marked())]
#[case::swmr_writer(true, SuperblockStatusFlags::swmr_write_marked())]
fn clear_status_flags_clears_marks_from_libhdf5_writers(
    #[case] swmr: bool,
    #[case] flags: SuperblockStatusFlags,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("interrupted.h5");
    let mut command = process::command(env!("CARGO_BIN_EXE_interrupted-writer")).unwrap();
    command
        .arg(&path)
        .arg(if swmr { "swmr" } else { "ordinary" });
    let child = Fixture::spawn(command, Duration::from_secs(30)).unwrap();
    let child = child
        .wait_ready(Readiness::try_from(b"HDF5_PURE_WRITER_READY\n".as_slice()).unwrap())
        .unwrap_or_else(|error| panic!("interrupted writer failed: {error}"));
    let output = child
        .interrupt()
        .unwrap_or_else(|error| panic!("writer cleanup failed: {error}"));
    assert!(!output.status.success(), "{output:?}");
    assert_eq!(output.stdout, b"HDF5_PURE_WRITER_READY\n");
    assert!(output.stderr.is_empty(), "{output:?}");
    assert_eq!(
        File::from_bytes(std::fs::read(&path).unwrap())
            .unwrap()
            .superblock()
            .consistency_flags(),
        flags,
    );
    let err = File::open(&path).unwrap_err();
    let Error::FileMarkedInUse(message) = err else {
        panic!("expected FileMarkedInUse, got {err:?}");
    };
    assert!(
        message.starts_with(&format!("{}: ", path.display())),
        "{message}"
    );
    assert!(
        message.contains(&format!("status flags {:#04x}", flags.bits())),
        "{message}"
    );
    assert!(message.contains("File::clear_status_flags"), "{message}");
    File::clear_status_flags(&path).unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.superblock().consistency_flags(),
        SuperblockStatusFlags::new()
    );
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), vec![1, 2, 3, 4]);
    let cleared = std::fs::read(&path).unwrap();
    File::clear_status_flags(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), cleared);
}
