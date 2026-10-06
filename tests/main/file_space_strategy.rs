//! File-space management strategy via the file-creation property list (the
//! `H5Pset_file_space_strategy` / `H5Pset_file_space_page_size` follow-on to
//! issue #21): the writer records the chosen strategy in a superblock-extension
//! File Space Info message, and the reader reads it back.

use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};
use hdf5_pure_core::__private::FileSpaceInfoFields;
use hdf5_pure_format::__private::DEFAULT_PAGE_SIZE;
use hdf5_pure_format::__private::DEFAULT_THRESHOLD;
use rstest::rstest;

use test_util::temp;

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr, 7)]
#[case::page(FileSpaceStrategy::Page, 7)]
#[case::aggr(FileSpaceStrategy::Aggr, DEFAULT_THRESHOLD)]
#[case::none(FileSpaceStrategy::None, DEFAULT_THRESHOLD)]
fn each_strategy_roundtrips(#[case] strategy: FileSpaceStrategy, #[case] threshold: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3, 4]);
    b.with_file_space_strategy(strategy, false, 7)
        .with_file_space_page_size(8192);
    b.write(&path).unwrap();

    let f = File::open(&path).unwrap();
    // The strategy and the page size round-trip.
    assert_eq!(f.file_space_strategy(), Some(strategy));
    let info = f.file_space_info().expect("file records a strategy");
    assert_eq!(info.strategy, strategy);
    assert!(!info.persist);
    assert_eq!(info.threshold, threshold);
    assert_eq!(info.page_size, 8192);
    assert!(info.manager_addrs.is_empty());
    let ext = f.superblock().superblock_extension_address;
    assert!(matches!(ext, Some(a) if a != u64::MAX));
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3, 4]
    );
}

#[rstest]
#[case::aggr(FileSpaceStrategy::Aggr)]
#[case::none(FileSpaceStrategy::None)]
fn a_strategy_without_a_free_space_manager_records_neither_persistence_nor_a_threshold(
    #[case] strategy: FileSpaceStrategy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strategy.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3, 4]);
    b.with_file_space_strategy(strategy, true, 7);
    b.write(&path).unwrap();

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.file_space_info(),
        Some(
            &FileSpaceInfoFields {
                strategy,
                persist: false,
                threshold: DEFAULT_THRESHOLD,
                page_size: DEFAULT_PAGE_SIZE,
                page_end_meta_threshold: 0,
                eoa_pre_fsm: u64::MAX,
                manager_addrs: Vec::new(),
            }
            .build()
        )
    );
}

#[test]
fn no_config_writes_no_extension() {
    let path = temp::temp_path("hdf5_pure_fss_default.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_f64_data(&[1.5, 2.5]);
    b.write(&path).unwrap();

    let f = File::open(&path).unwrap();
    // Default: no File Space Info message, matching the C library's behavior.
    assert_eq!(f.file_space_strategy(), None);
    assert!(f.file_space_info().is_none());
    assert_eq!(
        f.superblock().superblock_extension_address,
        Some(u64::MAX),
        "no extension is written when file space is unconfigured"
    );
}

#[test]
fn page_size_only_defaults_strategy() {
    let path = temp::temp_path("hdf5_pure_fss_pageonly.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[9]);
    b.with_file_space_page_size(2048);
    b.write(&path).unwrap();

    let f = File::open(&path).unwrap();
    let info = f
        .file_space_info()
        .expect("page size implies a strategy record");
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert_eq!(info.page_size, 2048);
    assert_eq!(info.threshold, 1); // default
}

#[test]
fn strategy_survives_userblock_and_empty_file() {
    // A userblock shifts base_address, so the extension's base-relative address
    // must still resolve; and a file with no datasets must still carry it.
    let path = temp::temp_path("hdf5_pure_fss_userblock.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[5, 6]);
    b.with_userblock(512)
        .with_file_space_strategy(FileSpaceStrategy::None, false, 1);
    b.write(&path).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::None));
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![5, 6]);

    let path = temp::temp_path("hdf5_pure_fss_empty.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Aggr, false, 1);
    b.write(&path).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::Aggr));
}

#[test]
fn survives_in_place_edit() {
    // Editing a strategy-bearing file in place must keep the strategy: the
    // superblock extension is preserved across the append-only commit, including
    // a delete that frees space (the extension pins end-of-file, so it is never
    // cut by truncation). Uses a non-paged strategy: a paged file cannot be edited
    // through the whole-file editor (issue #173 Phase 2), which is covered in
    // `tests/main/paged_mutation.rs`.
    let path = temp::temp_path("hdf5_pure_fss_edit.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, false, 1);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("added", |b| {
                b.with_f64_data(&vec![7.0; 512]);
            })
            .unwrap();
        s.commit().unwrap();
        s.root().delete("added").unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::FsmAggr));
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(f.dataset("added").is_err());
}

#[test]
fn persist_true_records_intent_on_a_fresh_file() {
    // A brand-new file has no free space, so persist = true records the persist
    // flag with no on-disk managers (matching the C library's brand-new persisted
    // file). A later File::open_rw that frees space fills the managers in.
    let path = temp::temp_path("hdf5_pure_fss_persist.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    let f = File::open(&path).unwrap();
    let info = f.file_space_info().expect("file records a strategy");
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert!(info.persist, "persist flag is recorded");
    // No free space yet: every manager slot is undefined and nothing is persisted.
    assert!(info.manager_addrs.iter().all(|&a| a == u64::MAX));
    assert!(f.persisted_free_space().unwrap().is_empty());
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![1, 2, 3]);
}
