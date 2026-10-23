//! Tests exact persisted free-space maps on the same closed file with both libraries.
//!
//! HDF5 1.10 and later run the equality assertions. Full paired snapshots use the
//! bundled build. Ownership accounting is a separate Rust diagnostic of live and free bytes.

#![cfg(feature = "__hdf5-1.10")]

use std::fs;
use std::io::Write as _;
use std::panic;
use std::path::Path;

use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5::{MajorErrorCode, MinorErrorCode};
use hdf5_pure::{File, FileBuilder, FileSpacePageSize, FileSpaceStrategy, UserblockSize};
use hdf5_sys::h5f;
use rstest::rstest;
use test_util_hdf5::file_space;
use test_util_hdf5::file_space_oracle::{
    self, ExtentError, FreeSpaceOracleCheckpoint, OracleError, SectionMap,
};
use test_util_hdf5::free_space_churn::{self, Base, Lifecycle, Strategy};
use test_util_hdf5::free_space_endurance;

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
#[case::none(FileSpaceStrategy::None)]
fn endurance_checkpoints_match_libhdf5(#[case] strategy: FileSpaceStrategy) {
    let case = format!(
        "endurance/{strategy:?}/persist={}/page=4096/threshold=1/base=0",
        strategy != FileSpaceStrategy::None
    );
    let mut observations = String::new();
    let mut steps = Vec::new();
    free_space_endurance::run_endurance_observed(strategy, |path, step| {
        steps.push(step.to_owned());
        let observation = observe(path, &case, step);
        if strategy != FileSpaceStrategy::None {
            assert!(
                observation.ownership.as_ref().unwrap().analysis_complete,
                "{case}/{step}"
            );
            assert!(
                observation
                    .ownership
                    .as_ref()
                    .unwrap()
                    .unclassified_space
                    .is_empty(),
                "{case}/{step}"
            );
            assert!(
                observation
                    .ownership
                    .as_ref()
                    .unwrap()
                    .trailing_slack_space
                    .is_empty(),
                "{case}/{step}"
            );
        } else {
            assert!(observation.rust.sections().is_empty(), "{case}/{step}");
        }
        observations.push_str(&observation.render());
        observations.push('\n');
    });
    assert_eq!(
        steps,
        (1..=16)
            .map(|i| format!("reopen-{}", i * 64))
            .chain(std::iter::once("final-close".to_owned()))
            .collect::<Vec<_>>()
    );
    snapshot(
        &format!("endurance_{strategy:?}").to_ascii_lowercase(),
        &observations,
    );
}

#[rstest]
#[case::continuous_fsm(Lifecycle::ContinuousOpen, Strategy::FsmAggr)]
#[case::continuous_page(Lifecycle::ContinuousOpen, Strategy::Page)]
#[case::final_fsm(Lifecycle::ReopenBeforeFinalCheck, Strategy::FsmAggr)]
#[case::final_page(Lifecycle::ReopenBeforeFinalCheck, Strategy::Page)]
#[case::frequent_fsm(Lifecycle::ReopenFrequently, Strategy::FsmAggr)]
fn churn_checkpoints_match_libhdf5(
    #[case] lifecycle: Lifecycle,
    #[case] strategy: Strategy,
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let case = format!("churn/{lifecycle:?}/{strategy:?}/{base:?}");
    let mut observations = String::new();
    free_space_churn::run_churn_observed(lifecycle, strategy, base, |path, step| {
        let observation = observe(path, &case, step);
        assert!(
            observation.ownership.as_ref().unwrap().analysis_complete,
            "{case}/{step}"
        );
        assert!(
            observation
                .ownership
                .as_ref()
                .unwrap()
                .unclassified_space
                .is_empty(),
            "{case}/{step}"
        );
        assert!(
            observation
                .ownership
                .as_ref()
                .unwrap()
                .trailing_slack_space
                .is_empty(),
            "{case}/{step}"
        );
        observations.push_str(&observation.render());
        observations.push('\n');
    });
    snapshot(
        &format!("churn_{lifecycle:?}_{strategy:?}_{base:?}").to_ascii_lowercase(),
        &observations,
    );
}

#[rstest]
#[case::fsm_persistent(FileSpaceStrategy::FsmAggr, true)]
#[case::page_persistent(FileSpaceStrategy::Page, true)]
#[case::fsm_nonpersistent(FileSpaceStrategy::FsmAggr, false)]
#[case::page_nonpersistent(FileSpaceStrategy::Page, false)]
#[case::aggr(FileSpaceStrategy::Aggr, false)]
#[case::none(FileSpaceStrategy::None, false)]
fn edits_by_both_libraries_preserve_matching_free_sections(
    #[case] strategy: FileSpaceStrategy,
    #[case] persist: bool,
    #[values(1, 128)] threshold: u64,
    #[values(0, 4096)] base: u64,
    #[values(false, true)] c_origin: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("round_trip.h5");
    let case = format!(
        "{strategy:?}/persist={persist}/page=4096/threshold={threshold}/base={base}/c_origin={c_origin}"
    );
    if c_origin {
        let c_strategy = match strategy {
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page => CStrategy::FreeSpaceManager {
                paged: strategy == FileSpaceStrategy::Page,
                persist,
                threshold,
            },
            FileSpaceStrategy::Aggr => CStrategy::PageAggregation,
            FileSpaceStrategy::None => CStrategy::None,
        };
        let file = hdf5::FileBuilder::new()
            .with_fapl(|p| p.libver_v110())
            .with_fcpl(|p| {
                p.file_space_strategy(c_strategy)
                    .file_space_page_size(4096)
                    .userblock(base)
            })
            .create(&path)
            .unwrap();
        for (name, value, len) in [("a", 1, 100), ("b", 2, 4096), ("c", 3, 100)] {
            file.new_dataset::<i32>()
                .shape([len])
                .create(name)
                .unwrap()
                .write_raw(&vec![value; len])
                .unwrap();
        }
        file.close().unwrap();
    } else {
        let mut builder = FileBuilder::new();
        builder
            .with_file_space_strategy(strategy, persist, threshold)
            .with_file_space_page_size(FileSpacePageSize::try_from(4096).unwrap());
        if base != 0 {
            builder.with_userblock(UserblockSize::try_from(base).unwrap());
        }
        for (name, value, len) in [("a", 1, 100), ("b", 2, 4096), ("c", 3, 100)] {
            builder
                .create_dataset(name)
                .with_i32_data(&vec![value; len]);
        }
        builder.write(&path).unwrap();
    }
    let mut observations = String::new();
    append_checkpoint(&mut observations, &path, &case, "created", persist);
    let file = File::open_rw(&path).unwrap();
    let freed = file_space::contiguous_extent(&file, "b");
    file.root().delete("b").unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    append_checkpoint(&mut observations, &path, &case, "rust-deletion", persist);
    for iteration in 0..3 {
        let file = File::open_rw(&path).unwrap();
        file.root()
            .create_dataset("replacement", |b| {
                b.with_i32_data(&vec![9; 3000 + iteration * 100]);
            })
            .unwrap();
        file.commit().unwrap();
        if persist && iteration == 0 {
            let allocated = file_space::contiguous_extent(&file, "replacement");
            assert!(
                allocated.start < freed.end && freed.start < allocated.end,
                "hole reuse; {case}"
            );
        }
        file.close().unwrap();
        append_checkpoint(
            &mut observations,
            &path,
            &case,
            &format!("rust-reuse-{iteration}"),
            persist,
        );
        let file = File::open_rw(&path).unwrap();
        file.root().delete("replacement").unwrap();
        file.commit().unwrap();
        file.close().unwrap();
        append_checkpoint(
            &mut observations,
            &path,
            &case,
            &format!("manager-rewrite-{iteration}"),
            persist,
        );
    }
    let file = hdf5::File::open_rw(&path).unwrap();
    let dataset = file
        .new_dataset::<i32>()
        .shape([3000])
        .create("from_c")
        .unwrap();
    dataset.write_raw(&vec![17; 3000]).unwrap();
    drop(dataset);
    file.close().unwrap();
    append_checkpoint(&mut observations, &path, &case, "c-reuse", persist);
    let file = hdf5::File::open_rw(&path).unwrap();
    file.unlink("from_c").unwrap();
    file.close().unwrap();
    append_checkpoint(&mut observations, &path, &case, "c-delete", persist);
    let reader = File::open(&path).unwrap();
    assert_eq!(
        reader.dataset("a").unwrap().read_i32().unwrap(),
        vec![1; 100]
    );
    assert_eq!(
        reader.dataset("c").unwrap().read_i32().unwrap(),
        vec![3; 100]
    );
    snapshot(
        &format!("round_trip_{strategy:?}_{persist}_{threshold}_{base}_{c_origin}")
            .to_ascii_lowercase(),
        &observations,
    );
}

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
fn records_stored_eof_separately_from_trailing_bytes(
    #[case] strategy: FileSpaceStrategy,
    #[values(0, 4096)] base: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trailing_bytes.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(strategy, true, 1);
    if base != 0 {
        builder.with_userblock(UserblockSize::try_from(base).unwrap());
    }
    for (name, value, len) in [("a", 1, 100), ("b", 2, 1024), ("c", 3, 100)] {
        builder
            .create_dataset(name)
            .with_i32_data(&vec![value; len]);
    }
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.root().delete("b").unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let reader = File::open(&path).unwrap();
    let logical = reader.superblock().eof_address;
    assert_eq!(reader.superblock().base_address.get(), base);
    reader.close().unwrap();
    file_space::assert_eof_matches_physical_size(&path);
    let slack = [0xA5; 257];
    {
        let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&slack).unwrap();
        file.sync_all().unwrap();
    }
    let slack_bytes = u64::try_from(slack.len()).unwrap();
    let physical = logical + slack_bytes;
    let error =
        panic::catch_unwind(|| file_space::assert_eof_matches_physical_size(&path)).unwrap_err();
    let message = error.downcast_ref::<String>().unwrap();
    assert_eq!(
        message,
        &format!(
            "assertion `left == right` failed: superblock EOF must match the physical file size\n  left: {logical}\n right: {physical}"
        ),
    );
    let case = format!("stored-eof/{strategy:?}/base={base}");
    let observation = observe(&path, &case, "trailing-bytes");
    assert_eq!(
        (
            observation.rust_logical_size,
            observation.c_file_size,
            observation.physical_file_size,
            observation.base,
        ),
        (logical, physical, physical, base),
    );
    let ownership = observation.ownership.as_ref().unwrap();
    assert!(ownership.analysis_complete);
    assert_eq!(ownership.logical_size, logical);
    assert_eq!(ownership.physical_size, physical);
    assert_eq!(ownership.trailing_slack_space, vec![(logical, slack_bytes)]);
    assert!(fs::read(&path).unwrap().ends_with(&slack));
    let reader = File::open(&path).unwrap();
    assert_eq!(
        reader.dataset("a").unwrap().read_i32().unwrap(),
        vec![1; 100]
    );
    assert_eq!(
        reader.dataset("c").unwrap().read_i32().unwrap(),
        vec![3; 100]
    );
    snapshot(
        &format!("stored_eof_{strategy:?}_{base}").to_ascii_lowercase(),
        &observation.render(),
    );
}

#[rstest]
#[case::different_addresses(vec![(16, 8)], vec![(32, 8)], 0, vec![(32, 8)], vec![(16, 8)])]
#[case::different_boundaries(vec![(16, 16)], vec![(16, 8), (24, 8)], 0, vec![(16, 8), (24, 8)], vec![(16, 16)])]
#[case::missing(vec![(16, 8)], vec![(16, 8), (32, 8)], 1, vec![(32, 8)], vec![])]
fn rejects_different_maps(
    #[case] rust: Vec<(u64, u64)>,
    #[case] c: Vec<(u64, u64)>,
    #[case] first: usize,
    #[case] missing: Vec<(u64, u64)>,
    #[case] extra: Vec<(u64, u64)>,
) {
    let rust = SectionMap::new("rust", rust, 128).unwrap();
    let c = SectionMap::new("c", c, 128).unwrap();
    let error = rust.compare(&c).unwrap_err();
    let OracleError::Mismatch {
        first: actual_first,
        missing: actual_missing,
        extra: actual_extra,
    } = error
    else {
        panic!("{error:?}");
    };
    assert_eq!(
        (actual_first, actual_missing, actual_extra),
        (first, missing, extra)
    );
}

#[rstest]
#[case::duplicate(vec![(16, 8), (16, 8)], 1, ExtentError::Overlap)]
#[case::overlap(vec![(16, 16), (24, 16)], 1, ExtentError::Overlap)]
#[case::zero_length(vec![(16, 0)], 0, ExtentError::ZeroLength)]
#[case::past_eoa(vec![(120, 16)], 0, ExtentError::OutOfBounds)]
#[case::overflow(vec![(u64::MAX, 1)], 0, ExtentError::AddressOverflow)]
fn rejects_invalid_sections(
    #[case] sections: Vec<(u64, u64)>,
    #[case] expected_index: usize,
    #[case] expected_reason: ExtentError,
) {
    let error = SectionMap::new("test", sections, 128).unwrap_err();
    let OracleError::Extent {
        source,
        index,
        reason,
        ..
    } = error
    else {
        panic!("{error:?}");
    };
    assert_eq!(
        (source, index, reason),
        ("test", expected_index, expected_reason)
    );
}

#[rstest]
#[case::short(2, 1)]
#[case::long(1, 2)]
fn rejects_incomplete_enumeration(#[case] expected: usize, #[case] actual: isize) {
    let error = file_space_oracle::enumeration_complete(expected, actual).unwrap_err();
    let OracleError::Enumeration {
        expected: got_expected,
        actual: got_actual,
    } = error
    else {
        panic!("{error:?}");
    };
    assert_eq!(
        (got_expected, got_actual),
        (expected, usize::try_from(actual).unwrap())
    );
}

#[test]
fn rejects_c_api_error() {
    let error = hdf5::sync::sync(|| {
        // SAFETY: The library lock serializes access. A null buffer and zero count write nothing.
        // The invalid identifier exercises the C API's argument error path.
        let count = unsafe {
            h5f::H5Fget_free_sections(-1, h5f::H5FD_MEM_DEFAULT, 0, std::ptr::null_mut())
        };
        file_space_oracle::enumeration_count(count).unwrap_err()
    });
    let OracleError::C { operation, source } = error else {
        panic!("{error:?}");
    };
    assert_eq!(operation, "H5Fget_free_sections");
    assert!(source.contains_major(MajorErrorCode::Args), "{source:?}");
    assert!(source.contains_minor(MinorErrorCode::BadType), "{source:?}");
}

/// Asserts exact section-map equality and snapshots the paired report with the bundled build.
///
/// # Panics
///
/// Panics if observation fails, section maps or C totals differ, or a bundled snapshot differs.
pub(super) fn checkpoint(path: &Path, case: &str, step: &str) {
    let observation = observe(path, case, step);
    snapshot(
        &format!("{case}_{step}").replace('/', "_"),
        &observation.render(),
    );
}

fn observe(path: &Path, case: &str, step: &str) -> FreeSpaceOracleCheckpoint {
    let observation = FreeSpaceOracleCheckpoint::observe(path, case, step)
        .unwrap_or_else(|error| panic!("{case}/{step}: {error:?}"));
    observation.assert_equal();
    observation
}

fn append_checkpoint(out: &mut String, path: &Path, case: &str, step: &str, persist: bool) {
    let observation = observe(path, case, step);
    if !persist {
        assert!(observation.rust.sections().is_empty(), "{case}/{step}");
    }
    out.push_str(&observation.render());
    out.push('\n');
}

fn snapshot(name: &str, report: &str) {
    #[cfg(feature = "__hdf5-bundled")]
    insta::assert_snapshot!(name, report);
    #[cfg(not(feature = "__hdf5-bundled"))]
    let _ = (name, report);
}
