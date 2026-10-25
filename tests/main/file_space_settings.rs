use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileCreateProperties;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::LibVer;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::UserblockSize;
use rstest::rstest;

#[test]
fn creation_properties_distinguish_absent_and_explicit_default_settings() {
    const ABSENT: FileCreateProperties = FileCreateProperties::new();
    const EXPLICIT: FileCreateProperties =
        ABSENT.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr));
    assert_eq!(ABSENT.file_space_settings(), None);
    assert_eq!(
        EXPLICIT.file_space_settings(),
        Some(FileSpaceSettings::default())
    );
    let build = |properties| {
        let mut builder = FileBuilder::new();
        builder.with_create_properties(properties);
        File::from_bytes(builder.finish().unwrap()).unwrap()
    };
    assert_eq!(build(ABSENT).file_space_info().unwrap(), None);
    assert_eq!(build(ABSENT).file_space_settings().unwrap(), None);
    assert_eq!(
        build(EXPLICIT).file_space_settings().unwrap(),
        EXPLICIT.file_space_settings()
    );
    assert!(build(EXPLICIT).file_space_info().unwrap().is_some());
}

#[rstest]
#[case::fsm_zero(FileSpaceStrategy::FsmAggr, false, 0, 512, 0)]
#[case::fsm_default(FileSpaceStrategy::FsmAggr, false, 1, 4096, 0)]
#[case::fsm_persistent(FileSpaceStrategy::FsmAggr, true, 128, 8192, 8192)]
#[case::fsm_irregular(FileSpaceStrategy::FsmAggr, true, 0, 3000, 512)]
#[case::page_zero(FileSpaceStrategy::Page, false, 0, 4096, 0)]
#[case::page_small(FileSpaceStrategy::Page, true, 1, 512, 512)]
#[case::page_persistent(FileSpaceStrategy::Page, true, 128, 8192, 8192)]
#[case::aggr(FileSpaceStrategy::Aggr, true, 0, 8192, 512)]
#[case::none(FileSpaceStrategy::None, true, 128, 8192, 512)]
fn settings_have_identical_creation_paths(
    #[case] strategy: FileSpaceStrategy,
    #[case] persist: bool,
    #[case] threshold: u64,
    #[case] page_size: u64,
    #[case] userblock: u64,
) {
    let page_size = FileSpacePageSize::try_from(page_size).unwrap();
    let settings = FileSpaceSettings::new(strategy)
        .with_persistence(persist)
        .with_threshold(threshold)
        .with_page_size(page_size);
    let properties = FileCreateProperties::new()
        .with_file_space_settings(settings)
        .with_userblock(UserblockSize::try_from(userblock).unwrap());
    assert_eq!(properties.file_space_settings(), Some(settings));
    let build = |from_properties: bool| {
        let mut builder = FileBuilder::new();
        if from_properties {
            builder.with_create_properties(properties);
        } else {
            builder.with_file_space_settings(settings);
            builder.with_userblock(properties.userblock());
        }
        builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
        builder
    };
    let bytes = build(false).finish().unwrap();
    assert_eq!(build(true).finish().unwrap(), bytes);
    let mut streamed = Vec::new();
    build(true).finish_to(&mut streamed).unwrap();
    assert_eq!(streamed, bytes);
    let mut legacy = FileBuilder::new();
    legacy.with_userblock(properties.userblock());
    legacy.with_file_space_strategy(strategy, persist, threshold);
    legacy.with_file_space_page_size(page_size);
    legacy.create_dataset("data").with_i32_data(&[1, 2, 3]);
    assert_eq!(legacy.finish().unwrap(), bytes);
    let file = File::from_bytes(bytes.clone()).unwrap();
    assert_eq!(file.file_space_settings().unwrap(), Some(settings));
    assert_eq!(file.dataset("data").unwrap().read_i32().unwrap(), [1, 2, 3]);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("settings.h5");
    build(false).write(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let file = File::open_streaming(&path).unwrap();
    assert_eq!(file.file_space_settings().unwrap(), Some(settings));
    drop(file);
    let path = directory.path().join("owned.h5");
    let file = File::create_with_options(&path, properties, FileAccessProperties::new()).unwrap();
    assert_eq!(file.file_space_settings().unwrap(), Some(settings));
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.file_space_settings().unwrap(), Some(settings));
}

#[test]
fn page_size_only_matches_explicit_default_strategy_settings() {
    let size = FileSpacePageSize::try_from(8192).unwrap();
    let mut legacy = FileBuilder::new();
    legacy.with_file_space_page_size(size);
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::default().with_page_size(size));
    assert_eq!(legacy.finish().unwrap(), builder.finish().unwrap());
}

#[test]
fn creation_properties_replace_and_clear_settings() {
    let original = FileSpaceSettings::new(FileSpaceStrategy::Page)
        .with_persistence(true)
        .with_page_size(FileSpacePageSize::try_from(8192).unwrap());
    let replacement =
        FileSpaceSettings::new(FileSpaceStrategy::None).with_page_size(FileSpacePageSize::MIN);
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(original);
    builder
        .with_create_properties(FileCreateProperties::new().with_file_space_settings(replacement));
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_eq!(file.file_space_settings().unwrap(), Some(replacement));
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(original);
    builder.with_create_properties(FileCreateProperties::new());
    assert_eq!(
        builder.finish().unwrap(),
        FileBuilder::new().finish().unwrap()
    );
}

#[rstest]
#[case::fsm(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
#[case::aggr(FileSpaceStrategy::Aggr)]
#[case::none(FileSpaceStrategy::None)]
fn settings_require_the_1_10_format(#[case] strategy: FileSpaceStrategy) {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy));
    builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    let error = builder.finish().unwrap_err();
    let Error::Format(FormatError::LibverTooOldForContent {
        content,
        needs,
        writing,
    }) = error
    else {
        panic!("expected LibverTooOldForContent, got {error:?}");
    };
    assert_eq!(
        (content, needs, writing),
        ("a file-space setting", "v1.10", "v1.8")
    );
}

#[test]
fn bounded_creation_rejects_nonpersistent_page_settings_before_writing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bounded.h5");
    let error = File::create_with_options(
        &path,
        FileCreateProperties::new()
            .with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page)),
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
    .unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected EditUnsupported, got {error:?}");
    };
    assert_eq!(
        reason,
        "bounded editing of PAGE files without persisted free space is unsupported; use MemoryStrategy::Auto or MemoryStrategy::Mirrored for staged edits"
    );
    assert!(!path.exists());
}
