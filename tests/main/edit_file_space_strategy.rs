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
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr, false, true)]
#[case::fsm_aggr_persisting(FileSpaceStrategy::FsmAggr, true, true)]
#[case::page_persisting(FileSpaceStrategy::Page, true, true)]
#[case::aggr(FileSpaceStrategy::Aggr, false, false)]
#[case::aggr_persisting(FileSpaceStrategy::Aggr, true, false)]
#[case::none(FileSpaceStrategy::None, false, false)]
#[case::none_persisting(FileSpaceStrategy::None, true, false)]
fn a_replacement_reuses_a_deleted_extent_only_under_a_free_space_manager(
    #[case] strategy: FileSpaceStrategy,
    #[case] persist: bool,
    #[case] reuses: bool,
) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    write_three_datasets(&path, strategy, persist);

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
    write_three_datasets(&path, strategy, false);

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
    write_three_datasets(&path, FileSpaceStrategy::FsmAggr, true);
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
    write_three_datasets(&path, FileSpaceStrategy::FsmAggr, false);
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

fn write_three_datasets(path: &Path, strategy: FileSpaceStrategy, persist: bool) {
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("b").with_i32_data(&[2; 400]);
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_strategy(strategy, persist, 1);
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
