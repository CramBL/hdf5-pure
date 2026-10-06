//! Free-space reuse in an edit under each file-space strategy.

use std::ops::Range;
use std::path::Path;

use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::Layout;
use hdf5_pure_core::__private::FileSpaceInfoFields;
use hdf5_pure_format::__private::FormatWidths;
use rstest::rstest;
use tempfile::tempdir;

use test_util::file_space_info;
use test_util::range;

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr, false, 1, true)]
#[case::fsm_aggr_below_the_extent(FileSpaceStrategy::FsmAggr, false, DELETED_LEN - 1, true)]
#[case::fsm_aggr_at_the_extent(FileSpaceStrategy::FsmAggr, false, DELETED_LEN, true)]
#[case::fsm_aggr_above_the_extent(FileSpaceStrategy::FsmAggr, false, DELETED_LEN + 1, false)]
#[case::fsm_aggr_persisting(FileSpaceStrategy::FsmAggr, true, 1, true)]
#[case::fsm_aggr_persisting_below_the_extent(
    FileSpaceStrategy::FsmAggr,
    true,
    DELETED_LEN - 1,
    true
)]
#[case::fsm_aggr_persisting_at_the_extent(FileSpaceStrategy::FsmAggr, true, DELETED_LEN, true)]
#[case::fsm_aggr_persisting_above_the_extent(
    FileSpaceStrategy::FsmAggr,
    true,
    DELETED_LEN + 1,
    false
)]
#[case::page_persisting(FileSpaceStrategy::Page, true, 1, true)]
#[case::aggr(FileSpaceStrategy::Aggr, false, 1, false)]
#[case::aggr_above_the_extent(FileSpaceStrategy::Aggr, false, DELETED_LEN + 1, false)]
#[case::aggr_persisting(FileSpaceStrategy::Aggr, true, 1, false)]
#[case::none(FileSpaceStrategy::None, false, 1, false)]
#[case::none_above_the_extent(FileSpaceStrategy::None, false, DELETED_LEN + 1, false)]
#[case::none_persisting(FileSpaceStrategy::None, true, 1, false)]
fn a_replacement_reuses_a_deleted_extent_only_where_a_free_space_manager_tracks_it(
    #[case] strategy: FileSpaceStrategy,
    #[case] persist: bool,
    #[case] threshold: u64,
    #[case] reuses: bool,
) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    write_three_datasets(&path, strategy, persist, threshold);

    let file = File::open_rw(&path).unwrap();
    let deleted = contiguous_extent(&file, "b");
    file.root().delete("b").unwrap();
    file.commit().unwrap();
    file.root()
        .create_dataset("d", |b| {
            b.with_i32_data(&[4; 300]);
        })
        .unwrap();
    file.commit().unwrap();
    let placed = contiguous_extent(&file, "d");

    assert_eq!(
        range::overlaps(&deleted, &placed),
        reuses,
        "{deleted:?}, {placed:?}"
    );
}

#[rstest]
#[case::aggr(FileSpaceStrategy::Aggr)]
#[case::none(FileSpaceStrategy::None)]
fn a_deletion_leaves_no_reusable_free_space_without_a_free_space_manager(
    #[case] strategy: FileSpaceStrategy,
) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    write_three_datasets(&path, strategy, false, 1);

    let file = File::open_rw(&path).unwrap();
    file.root().delete("b").unwrap();
    file.commit().unwrap();

    assert_eq!(file.space_accounting().unwrap().reusable_free_space, []);
}

#[rstest]
#[case::aggr(FileSpaceStrategy::Aggr)]
#[case::none(FileSpaceStrategy::None)]
fn persisted_free_space_is_not_reused_under_a_strategy_without_a_free_space_manager(
    #[case] strategy: FileSpaceStrategy,
) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    // An FsmAggr file's persisted free space, relabeled with `strategy` below.
    write_three_datasets(&path, FileSpaceStrategy::FsmAggr, true, 1);
    let deleted = {
        let file = File::open_rw(&path).unwrap();
        let deleted = contiguous_extent(&file, "b");
        file.root().delete("b").unwrap();
        file.commit().unwrap();
        deleted
    };
    let (extension, info, free) = {
        let file = File::open(&path).unwrap();
        (
            file.superblock().superblock_extension_address.unwrap(),
            file.file_space_info().unwrap().clone(),
            file.persisted_free_space().unwrap(),
        )
    };
    let relabeled = FileSpaceInfoFields {
        strategy,
        persist: info.persist,
        threshold: info.threshold,
        page_size: info.page_size,
        page_end_meta_threshold: info.page_end_meta_threshold,
        eoa_pre_fsm: info.eoa_pre_fsm,
        manager_addrs: info.manager_addrs,
    }
    .build();
    file_space_info::replace_message(
        &path,
        extension,
        &hdf5_pure_format::__private::serialize_file_space_info(
            FormatWidths::from_sizes(8, 8).unwrap(),
            &relabeled,
        )
        .unwrap(),
    );

    let placed = {
        let file = File::open_rw(&path).unwrap();
        file.root()
            .create_dataset("d", |b| {
                b.with_i32_data(&[4; 300]);
            })
            .unwrap();
        file.commit().unwrap();
        contiguous_extent(&file, "d")
    };

    assert!(
        !range::overlaps(&deleted, &placed),
        "{deleted:?}, {placed:?}"
    );
    let file = File::open(&path).unwrap();
    assert_eq!(
        (file.file_space_info(), file.persisted_free_space().unwrap()),
        (Some(&relabeled), free)
    );
}

#[test]
fn a_file_whose_strategy_cannot_be_read_reuses_no_deleted_extent() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    write_three_datasets(&path, FileSpaceStrategy::FsmAggr, false, 1);
    let extension = File::open(&path)
        .unwrap()
        .superblock()
        .superblock_extension_address
        .unwrap();
    file_space_info::replace_message(&path, extension, &[UNKNOWN_VERSION]);

    let file = File::open_rw(&path).unwrap();
    let deleted = contiguous_extent(&file, "b");
    file.root().delete("b").unwrap();
    file.commit().unwrap();
    file.root()
        .create_dataset("d", |b| {
            b.with_i32_data(&[4; 300]);
        })
        .unwrap();
    file.commit().unwrap();
    let placed = contiguous_extent(&file, "d");

    assert!(
        !range::overlaps(&deleted, &placed),
        "{deleted:?}, {placed:?}"
    );
}

#[rstest]
fn a_deleted_extent_is_tracked_from_the_threshold_up(
    #[values(false, true)] persist: bool,
    #[values(DELETED_LEN - 1, DELETED_LEN, DELETED_LEN + 1)] threshold: u64,
) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    write_three_datasets(&path, FileSpaceStrategy::FsmAggr, persist, threshold);

    let (deleted, session) = {
        let file = File::open_rw(&path).unwrap();
        let deleted = contiguous_extent(&file, "b");
        file.root().delete("b").unwrap();
        file.commit().unwrap();
        (
            deleted,
            file.space_accounting().unwrap().reusable_free_space,
        )
    };
    let reopened = File::open(&path).unwrap().persisted_free_space().unwrap();

    let tracked = if threshold <= DELETED_LEN {
        DELETED_LEN
    } else {
        0
    };
    assert_eq!(
        (
            range::covered_len(&session, &deleted),
            range::covered_len(&reopened, &deleted)
        ),
        (tracked, if persist { tracked } else { 0 })
    );
}

#[rstest]
#[case::alone(vec![vec!["c"]], 0)]
#[case::beside_a_tracked_extent(vec![vec!["b"], vec!["c"]], SUB_THRESHOLD_LEN)]
#[case::above_a_tracked_extent_in_one_commit(vec![vec!["b", "c"]], SUB_THRESHOLD_LEN)]
#[case::below_a_tracked_extent_in_one_commit(vec![vec!["e", "g"]], NEAR_THRESHOLD_LEN)]
#[case::beside_a_sub_threshold_extent(vec![vec!["c"], vec!["e"]], 0)]
#[case::beside_a_sub_threshold_extent_in_one_commit(vec![vec!["c", "e"]], 0)]
fn a_sub_threshold_extent_is_tracked_only_beside_a_tracked_extent(
    #[values(false, true)] persist: bool,
    #[case] commits: Vec<Vec<&str>>,
    #[case] tracked: u64,
) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("b")
        .with_i32_data(&[2; THRESHOLD_ELEMENTS]);
    b.create_dataset("c")
        .with_i32_data(&[3; SUB_THRESHOLD_ELEMENTS]);
    b.create_dataset("e")
        .with_i32_data(&[4; NEAR_THRESHOLD_ELEMENTS]);
    b.create_dataset("g")
        .with_i32_data(&[5; THRESHOLD_ELEMENTS]);
    b.create_dataset("f").with_i32_data(&[6; 100]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, persist, THRESHOLD);
    b.write(&path).unwrap();

    let (extents, session) = {
        let file = File::open_rw(&path).unwrap();
        let extents = ["b", "c", "e", "g"].map(|path| contiguous_extent(&file, path));
        for paths in commits {
            for path in paths {
                file.root().delete(path).unwrap();
            }
            file.commit().unwrap();
        }
        (
            extents,
            file.space_accounting().unwrap().reusable_free_space,
        )
    };
    let reopened = File::open(&path).unwrap().persisted_free_space().unwrap();

    let [b, c, e, g] = &extents;
    assert_eq!((b.end, c.end, e.end), (c.start, e.start, g.start));
    let covered = |sections: &[(u64, u64)]| -> u64 {
        [c, e]
            .into_iter()
            .map(|extent| range::covered_len(sections, extent))
            .sum()
    };
    assert_eq!(
        (covered(&session), covered(&reopened)),
        (tracked, if persist { tracked } else { 0 })
    );
}

#[test]
fn sub_threshold_extents_that_end_the_file_are_given_back_in_one_commit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("b")
        .with_i32_data(&[2; THRESHOLD_ELEMENTS]);
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, false, THRESHOLD);
    b.write(&path).unwrap();

    let file = File::open_rw(&path).unwrap();
    file.root().delete("b").unwrap();
    file.commit().unwrap();
    file.root()
        .create_dataset("w", |b| {
            b.with_i32_data(&[8; 700]);
        })
        .unwrap();
    file.commit().unwrap();
    for path in ["x", "y"] {
        file.root()
            .create_dataset(path, |b| {
                b.with_i32_data(&[7; NEAR_THRESHOLD_ELEMENTS]);
            })
            .unwrap();
    }
    file.commit().unwrap();
    let (x, y) = (contiguous_extent(&file, "x"), contiguous_extent(&file, "y"));
    let len = file.space_accounting().unwrap().logical_size;
    file.root().delete("x").unwrap();
    file.root().delete("y").unwrap();
    file.commit().unwrap();

    assert_eq!(
        (x.end, y.end, file.space_accounting().unwrap().logical_size),
        (y.start, len, x.start)
    );
}

fn write_three_datasets(path: &Path, strategy: FileSpaceStrategy, persist: bool, threshold: u64) {
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("b").with_i32_data(&[2; DELETED_ELEMENTS]);
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_strategy(strategy, persist, threshold);
    b.write(path).unwrap();
}

fn contiguous_extent(file: &File, path: &str) -> Range<u64> {
    let layout = file.dataset(path).unwrap().layout().unwrap();
    let Layout::Contiguous {
        address: Some(address),
        size,
    } = layout
    else {
        panic!("expected allocated contiguous storage, got {layout:?}");
    };
    address..address + size
}

const UNKNOWN_VERSION: u8 = 0xff;
const DELETED_ELEMENTS: usize = 400;
const DELETED_LEN: u64 = (DELETED_ELEMENTS * size_of::<i32>()) as u64;
const THRESHOLD_ELEMENTS: usize = 1024;
const THRESHOLD: u64 = (THRESHOLD_ELEMENTS * size_of::<i32>()) as u64;
const SUB_THRESHOLD_ELEMENTS: usize = 256;
const SUB_THRESHOLD_LEN: u64 = (SUB_THRESHOLD_ELEMENTS * size_of::<i32>()) as u64;
const NEAR_THRESHOLD_ELEMENTS: usize = THRESHOLD_ELEMENTS - 1;
const NEAR_THRESHOLD_LEN: u64 = (NEAR_THRESHOLD_ELEMENTS * size_of::<i32>()) as u64;
