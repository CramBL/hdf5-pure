#![cfg(feature = "__hdf5-1.10")]
//! Successive native reads while a pure SWMR writer remains open.
//!
//! Each append state is observed by a fresh reader process. After the writer
//! closes, another process uses normal read mode. The file-locking override
//! applies only to the child processes.

use std::path::Path;
use std::time::Duration;

use hdf5_pure::{File, SuperblockStatusFlags};

use test_util::process::{self, Fixture};
use test_util::superblock;
use test_util::temp;
use test_util_hdf5::dataset::Unlimited;

use crate::test_util::swmr_reader::{self, Mode, Observation};

#[test]
fn successive_native_readers_observe_pure_appends() {
    let path = temp::temp_path("swmr-successive.h5");
    Unlimited::new("d", &(0..5).collect::<Vec<i32>>(), 1).pure_create(&path);
    let writer = File::open_swmr_writer(&path).unwrap();
    writer
        .dataset("d")
        .unwrap()
        .append(&(5..50).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(
        superblock::consistency_flags(&path),
        SuperblockStatusFlags::swmr_write_marked().bits()
    );
    assert_observation(&path, Mode::Swmr, 50);

    writer
        .dataset("d")
        .unwrap()
        .append(&(50..200).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(
        superblock::consistency_flags(&path),
        SuperblockStatusFlags::swmr_write_marked().bits()
    );
    assert_observation(&path, Mode::Swmr, 200);

    writer.close().unwrap();
    assert_eq!(
        superblock::consistency_flags(&path),
        SuperblockStatusFlags::new().bits()
    );
    assert_observation(&path, Mode::Normal, 200);
}

fn assert_observation(path: &Path, mode: Mode, end: i32) {
    let mut command = process::command(env!("CARGO_BIN_EXE_swmr-reader")).unwrap();
    command
        .arg(path)
        .arg(serde_json::to_string(&mode).unwrap())
        .env("HDF5_USE_FILE_LOCKING", "FALSE");
    let output = Fixture::spawn(command, Duration::from_secs(20))
        .unwrap_or_else(|error| panic!("native reader launch failed: {error}"))
        .finish()
        .unwrap_or_else(|error| panic!("native reader failed: {error}"));
    let observation = swmr_reader::observe(mode, output)
        .unwrap_or_else(|error| panic!("native reader protocol failed: {error}"));
    assert_eq!(
        observation,
        Observation {
            mode,
            shape: [usize::try_from(end).unwrap()],
            values: (0..end).collect(),
        },
    );
}

#[rstest::rstest]
#[case::missing_path(&[], "missing fixture path")]
#[case::missing_mode(&["unused"], "missing reader mode")]
#[case::unknown_mode(&["unused", r#""Other""#], "unknown variant")]
#[case::extra_argument(&["unused", r#""Swmr""#, "extra"], "unexpected fixture argument")]
fn fixture_rejects_malformed_arguments(#[case] args: &[&str], #[case] expected: &str) {
    let mut command = process::command(env!("CARGO_BIN_EXE_swmr-reader")).unwrap();
    command.args(args);
    let output = Fixture::spawn(command, Duration::from_secs(20))
        .unwrap()
        .finish()
        .unwrap_or_else(|error| panic!("native reader failed: {error}"));
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(expected),
        "{output:?}"
    );
}
