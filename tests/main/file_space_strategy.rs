//! File-space management strategy via the file-creation property list (the
//! `H5Pset_file_space_strategy` / `H5Pset_file_space_page_size` follow-on to
//! issue #21): the writer records the chosen strategy in a superblock-extension
//! File Space Info message, and the reader reads it back.

use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::StoredAddress;
use hdf5_pure::UserblockSize;
use hdf5_pure_core::__private::FileSpaceInfoFields;
use hdf5_pure_format::__private::DEFAULT_THRESHOLD;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
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
    b.with_file_space_settings(
        FileSpaceSettings::new(strategy)
            .with_threshold(7)
            .with_page_size(FileSpacePageSize::try_from(8192).unwrap()),
    );
    b.write(&path).unwrap();

    let f = File::open(&path).unwrap();
    // The strategy and the page size round-trip.
    assert_eq!(
        f.file_space_settings()
            .unwrap()
            .map(|settings| settings.strategy()),
        Some(strategy)
    );
    let info = f
        .file_space_info()
        .unwrap()
        .expect("file records a strategy");
    assert_eq!(info.strategy, strategy);
    assert!(!info.persist);
    assert_eq!(info.threshold, threshold);
    assert_eq!(info.page_size.get(), 8192);
    assert!(info.manager_addrs.is_empty());
    let ext = f.superblock().superblock_extension_address;
    let offset_width = f.superblock().offset_size;
    assert!(matches!(ext, Some(a) if !a.is_undefined(offset_width)));
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
    b.with_file_space_settings(
        FileSpaceSettings::new(strategy)
            .with_persistence(true)
            .with_threshold(7),
    );
    b.write(&path).unwrap();

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.file_space_info().unwrap(),
        Some(
            FileSpaceInfoFields {
                strategy,
                persist: false,
                threshold: DEFAULT_THRESHOLD,
                page_size: FileSpacePageSize::DEFAULT,
                page_end_meta_threshold: 0,
                eoa_pre_fsm: Some(hdf5_pure::DecodedStoredAddress::Undefined),
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
    assert_eq!(
        f.file_space_settings()
            .unwrap()
            .map(|settings| settings.strategy()),
        None
    );
    assert!(f.file_space_info().unwrap().is_none());
    assert_eq!(
        f.superblock()
            .superblock_extension_address
            .map(StoredAddress::get),
        Some(u64::MAX),
        "no extension is written when file space is unconfigured"
    );
}

#[test]
fn fsm_settings_record_a_custom_page_size() {
    let path = temp::temp_path("hdf5_pure_fss_pageonly.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[9]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_page_size(FileSpacePageSize::try_from(2048).unwrap()),
    );
    b.write(&path).unwrap();

    let f = File::open(&path).unwrap();
    let info = f
        .file_space_info()
        .unwrap()
        .expect("page size implies a strategy record");
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert_eq!(info.page_size.get(), 2048);
    assert_eq!(info.threshold, 1); // default
}

#[test]
fn strategy_survives_userblock_and_empty_file() {
    // A userblock shifts base_address, so the extension's base-relative address
    // must still resolve; and a file with no datasets must still carry it.
    let path = temp::temp_path("hdf5_pure_fss_userblock.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[5, 6]);
    b.with_userblock(UserblockSize::MIN)
        .with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::None));
    b.write(&path).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.file_space_settings()
            .unwrap()
            .map(|settings| settings.strategy()),
        Some(FileSpaceStrategy::None)
    );
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![5, 6]);

    let path = temp::temp_path("hdf5_pure_fss_empty.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Aggr));
    b.write(&path).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.file_space_settings()
            .unwrap()
            .map(|settings| settings.strategy()),
        Some(FileSpaceStrategy::Aggr)
    );
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
    b.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr));
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
    assert_eq!(
        f.file_space_settings()
            .unwrap()
            .map(|settings| settings.strategy()),
        Some(FileSpaceStrategy::FsmAggr)
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    test_util_hdf5::assertion::path_not_found(f.dataset("added").unwrap_err(), "added");
}

#[test]
fn persist_true_records_reusable_spare_on_a_fresh_file() {
    let path = temp::temp_path("hdf5_pure_fss_persist.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    b.write(&path).unwrap();

    let f = File::open(&path).unwrap();
    let info = f
        .file_space_info()
        .unwrap()
        .expect("file records a strategy");
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert!(info.persist, "persist flag is recorded");
    assert!(
        info.manager_addrs
            .iter()
            .any(|&a| matches!(a, hdf5_pure::DecodedStoredAddress::Defined(_)))
    );
    let report = f.persisted_free_sections().unwrap().unwrap();
    assert!(report.free_sections().section_count() > 0);
    let bytes = std::fs::read(&path).unwrap();
    let widths = FormatWidths {
        offsets: f.superblock().offset_size,
        lengths: f.superblock().length_size,
    };
    let expected_total = report
        .managers()
        .iter()
        .map(|manager| {
            let address = usize::try_from(manager.address().get()).unwrap();
            let header = FreeSpaceManagerHeader::parse(widths, &bytes[address..]).unwrap();
            let expected = hdf5_pure_format::__private::free_space_manager_header_len(widths)
                + header.section_list_allocated();
            assert_eq!(manager.metadata_bytes(), expected);
            expected
        })
        .sum::<u64>();
    assert_eq!(report.manager_metadata_bytes(), expected_total);
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![1, 2, 3]);
}
