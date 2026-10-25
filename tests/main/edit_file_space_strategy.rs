//! Free-space reuse in an edit under each file-space strategy.

use std::path::Path;

use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::FreeSectionCoordinates;
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::__private::FileSpaceInfoFields;
use hdf5_pure_format::__private::FormatWidths;
use rstest::rstest;

use test_util::file_space_info;
use test_util::range;
use test_util::widths::Widths;
use test_util_hdf5::file_space;
use test_util_hdf5::storage;

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
#[case::page_persisting_below_the_extent(FileSpaceStrategy::Page, true, DELETED_LEN - 1, true)]
#[case::page_persisting_at_the_extent(FileSpaceStrategy::Page, true, DELETED_LEN, true)]
#[case::page_persisting_above_the_extent(FileSpaceStrategy::Page, true, DELETED_LEN + 1, false)]
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
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    file_space::write_strategy_triplet(&path, strategy, persist, threshold);

    let file = File::open_rw(&path).unwrap();
    let deleted = file_space::contiguous_extent(&file, "b");
    file.root().delete("b").unwrap();
    file.commit().unwrap();
    file.root()
        .create_dataset("d", |b| {
            b.with_i32_data(&[4; 400]);
        })
        .unwrap();
    file.commit().unwrap();
    let placed = file_space::contiguous_extent(&file, "d");

    assert_eq!(
        range::overlaps(&deleted, &placed),
        reuses,
        "{deleted:?}, {placed:?}"
    );
}

#[rstest]
#[case::aggr(FileSpaceStrategy::Aggr, true)]
#[case::none(FileSpaceStrategy::None, false)]
fn a_deletion_discards_its_extent_without_a_free_space_manager(
    #[case] strategy: FileSpaceStrategy,
    #[case] has_aggregate: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    file_space::write_strategy_triplet(&path, strategy, false, 1);

    let file = File::open_rw(&path).unwrap();
    let deleted = file_space::contiguous_extent(&file, "b");
    file.root().delete("b").unwrap();
    file.commit().unwrap();

    let free = file.reusable_free_sections().unwrap();
    assert_eq!(free.section_count() > 0, has_aggregate);
    assert!(
        free.sections()
            .iter()
            .all(|section| !range::overlaps(&deleted, &(section.address()..section.end_address()))),
        "{deleted:?}, {free:?}"
    );
}

#[rstest]
#[case::aggr(FileSpaceStrategy::Aggr)]
#[case::none(FileSpaceStrategy::None)]
fn persisted_free_space_is_not_reused_under_a_strategy_without_a_free_space_manager(
    #[case] strategy: FileSpaceStrategy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    // An FsmAggr file's persisted free space, relabeled with `strategy` below.
    file_space::write_strategy_triplet(&path, FileSpaceStrategy::FsmAggr, true, 1);
    let deleted = {
        let file = File::open_rw(&path).unwrap();
        let deleted = file_space::contiguous_extent(&file, "b");
        file.root().delete("b").unwrap();
        file.commit().unwrap();
        deleted
    };
    let (extension, info, free) = {
        let file = File::open(&path).unwrap();
        (
            file.superblock()
                .base_address
                .absolute(file.superblock().superblock_extension_address.unwrap())
                .unwrap(),
            file.file_space_info().unwrap().unwrap(),
            file.persisted_free_sections().unwrap().unwrap(),
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
        &hdf5_pure_format::__private::serialize_file_space_info_v1(
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
        file_space::contiguous_extent(&file, "d")
    };

    assert!(
        !range::overlaps(&deleted, &placed),
        "{deleted:?}, {placed:?}"
    );
    let file = File::open(&path).unwrap();
    assert_eq!(file.file_space_info().unwrap(), Some(relabeled));
    assert_eq!(
        file.persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections(),
        free.free_sections()
    );
}

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr)]
#[case::paged(FileSpaceStrategy::Page)]
fn unreadable_file_space_strategy_is_rejected_before_mutation(#[case] strategy: FileSpaceStrategy) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    file_space::write_strategy_triplet(&path, strategy, false, 1);
    let extension = {
        let file = File::open(&path).unwrap();
        let superblock = file.superblock();
        superblock
            .base_address
            .absolute(superblock.superblock_extension_address.unwrap())
            .unwrap()
    };
    file_space_info::replace_message(&path, extension, &[UNKNOWN_VERSION]);
    let original = std::fs::read(&path).unwrap();
    let error = File::open_rw(&path).unwrap_err();
    let Error::Format(FormatError::UnsupportedFileSpaceInfoVersion(version)) = error else {
        panic!("expected UnsupportedFileSpaceInfoVersion, got {error:?}");
    };
    assert_eq!(version, UNKNOWN_VERSION);
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[test]
fn editing_a_non_persisting_version_0_file_reuses_a_deleted_extent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v0_non_persistent.h5");
    file_space::write_strategy_triplet(&path, FileSpaceStrategy::FsmAggr, false, 1);
    let extension = {
        let file = File::open(&path).unwrap();
        let superblock = file.superblock();
        superblock
            .base_address
            .absolute(superblock.superblock_extension_address.unwrap())
            .unwrap()
    };
    file_space_info::replace_message(
        &path,
        extension,
        &file_space_info::version_0(Widths::EIGHT, file_space_info::ALL, 1, &[]),
    );

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.file_space_info().unwrap(),
        Some(
            FileSpaceInfoFields {
                strategy: FileSpaceStrategy::FsmAggr,
                persist: false,
                threshold: 1,
                page_size: FileSpacePageSize::DEFAULT,
                page_end_meta_threshold: 0,
                eoa_pre_fsm: None,
                manager_addrs: Vec::new(),
            }
            .build()
        )
    );
    drop(file);

    let file = File::open_rw(&path).unwrap();
    let deleted = file_space::contiguous_extent(&file, "b");
    file.root().delete("b").unwrap();
    file.commit().unwrap();
    file.root()
        .create_dataset("d", |b| {
            b.with_i32_data(&[4; 300]);
        })
        .unwrap();
    file.commit().unwrap();
    let placed = file_space::contiguous_extent(&file, "d");

    assert!(
        range::overlaps(&deleted, &placed),
        "{deleted:?}, {placed:?}"
    );
    drop(file);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), vec![4; 300]);
    assert_eq!(
        file.file_space_info().unwrap(),
        Some(
            FileSpaceInfoFields {
                strategy: FileSpaceStrategy::FsmAggr,
                persist: false,
                threshold: 1,
                page_size: FileSpacePageSize::DEFAULT,
                page_end_meta_threshold: 0,
                eoa_pre_fsm: None,
                manager_addrs: Vec::new(),
            }
            .build()
        )
    );
}

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr, false)]
#[case::fsm_aggr_persisting(FileSpaceStrategy::FsmAggr, true)]
#[case::page_persisting(FileSpaceStrategy::Page, true)]
fn a_deleted_extent_is_tracked_from_the_threshold_up(
    #[case] strategy: FileSpaceStrategy,
    #[case] persist: bool,
    #[values(DELETED_LEN - 1, DELETED_LEN, DELETED_LEN + 1)] threshold: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    file_space::write_strategy_triplet(&path, strategy, persist, threshold);

    let (deleted, session) = {
        let file = File::open_rw(&path).unwrap();
        let deleted = file_space::contiguous_extent(&file, "b");
        file.root().delete("b").unwrap();
        file.commit().unwrap();
        (deleted, file.reusable_free_sections().unwrap())
    };
    let reopened = persisted_or_reused_sections(&path, persist);

    let tracked = if threshold <= DELETED_LEN {
        DELETED_LEN
    } else {
        0
    };
    assert_eq!(
        (
            test_util_hdf5::covered_free_bytes!(
                &session,
                &deleted,
                FreeSectionCoordinates::Absolute
            ),
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
    let dir = tempfile::tempdir().unwrap();
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
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(persist)
            .with_threshold(THRESHOLD),
    );
    b.write(&path).unwrap();

    let (extents, session) = {
        let file = File::open_rw(&path).unwrap();
        let extents = ["b", "c", "e", "g"].map(|path| file_space::contiguous_extent(&file, path));
        for paths in commits {
            for path in paths {
                file.root().delete(path).unwrap();
            }
            file.commit().unwrap();
        }
        (extents, file.reusable_free_sections().unwrap())
    };
    let reopened = persisted_or_reused_sections(&path, persist);

    let [b, c, e, g] = &extents;
    assert_eq!((b.end, c.end, e.end), (c.start, e.start, g.start));
    let covered = |sections: &[(u64, u64)]| -> u64 {
        [c, e]
            .into_iter()
            .map(|extent| range::covered_len(sections, extent))
            .sum()
    };
    assert_eq!(
        (
            [c, e]
                .into_iter()
                .map(|extent| test_util_hdf5::covered_free_bytes!(
                    &session,
                    extent,
                    FreeSectionCoordinates::Absolute
                ))
                .sum::<u64>(),
            covered(&reopened)
        ),
        (tracked, if persist { tracked } else { 0 })
    );
}

fn persisted_or_reused_sections(path: &Path, persist: bool) -> Vec<(u64, u64)> {
    let report = File::open(path).unwrap().persisted_free_sections().unwrap();
    assert_eq!(report.is_some(), persist);
    let mut sections = report
        .as_ref()
        .map(|report| {
            report
                .free_sections()
                .sections()
                .iter()
                .map(|section| {
                    assert_eq!(section.coordinates(), FreeSectionCoordinates::Stored);
                    (section.address(), section.length().get())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if persist {
        let bytes = std::fs::read(path).unwrap();
        let file = File::open(path).unwrap();
        sections.extend(storage::file_metadata_extents(&file, &bytes));
    }
    sections
}

#[test]
fn sub_threshold_extents_that_end_the_file_are_given_back_in_one_commit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("b")
        .with_i32_data(&[2; THRESHOLD_ELEMENTS]);
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_threshold(THRESHOLD),
    );
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
    let stats = file.reusable_space_stats().unwrap();
    let sections = file.reusable_free_sections().unwrap();
    let retained_end = sections
        .sections()
        .iter()
        .find(|section| section.end_address() == stats.logical_size())
        .unwrap()
        .address();
    for path in ["x", "y"] {
        file.root()
            .create_dataset(path, |b| {
                b.with_i32_data(&[7; NEAR_THRESHOLD_ELEMENTS]);
            })
            .unwrap();
    }
    file.commit().unwrap();
    let (x, y) = (
        file_space::contiguous_extent(&file, "x"),
        file_space::contiguous_extent(&file, "y"),
    );
    let len = file.reusable_space_stats().unwrap().logical_size();
    file.root().delete("x").unwrap();
    file.root().delete("y").unwrap();
    file.commit().unwrap();

    assert_eq!(x.start - retained_end, 1977);
    assert_eq!(
        (
            x.end,
            y.end,
            file.reusable_space_stats().unwrap().logical_size()
        ),
        (y.start, len, retained_end)
    );
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.file_size(), retained_end);
    assert_eq!(file.root().datasets().unwrap(), ["a", "c", "w"]);
    assert_eq!(file.dataset("a").unwrap().read_i32().unwrap(), [1; 100]);
    assert_eq!(file.dataset("c").unwrap().read_i32().unwrap(), [3; 100]);
    assert_eq!(file.dataset("w").unwrap().read_i32().unwrap(), [8; 700]);
}

const UNKNOWN_VERSION: u8 = 0xff;
const DELETED_LEN: u64 = file_space::STRATEGY_DELETED_LEN;
const THRESHOLD_ELEMENTS: usize = 1024;
const THRESHOLD: u64 = (THRESHOLD_ELEMENTS * size_of::<i32>()) as u64;
const SUB_THRESHOLD_ELEMENTS: usize = 256;
const SUB_THRESHOLD_LEN: u64 = (SUB_THRESHOLD_ELEMENTS * size_of::<i32>()) as u64;
const NEAR_THRESHOLD_ELEMENTS: usize = THRESHOLD_ELEMENTS - 1;
const NEAR_THRESHOLD_LEN: u64 = (NEAR_THRESHOLD_ELEMENTS * size_of::<i32>()) as u64;
