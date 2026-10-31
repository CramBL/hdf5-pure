use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::UserblockSize;
use rstest::rstest;

use test_util_hdf5::aggregation;
use test_util_hdf5::aggregation::Object;
use test_util_hdf5::aggregation::Step;
use test_util_hdf5::aggregation::Workload;
use test_util_hdf5::file_space;
use test_util_hdf5::storage;

#[rstest]
#[case::mirror(MemoryStrategy::Mirrored, 0)]
#[case::bounded(MemoryStrategy::Bounded, 0)]
#[case::userblock(MemoryStrategy::Mirrored, 512)]
fn persistent_creation_recovers_settled_spare_and_reuses_it_after_reopen(
    #[case] backing: MemoryStrategy,
    #[case] userblock: u64,
    #[values(1, 4_096, u64::MAX)] threshold: u64,
) {
    let has_manager = threshold == 1;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("persistent.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(threshold),
    );
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.create_dataset("values").with_i32_data(&[17; 16]);
    builder.write(&path).unwrap();
    let file = File::open(&path).unwrap();
    let free = file.persisted_free_sections().unwrap().unwrap();
    assert_eq!(free.free_sections().section_count() > 0, has_manager);
    drop(file);
    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
    assert_eq!(
        !file
            .file_space_info()
            .unwrap()
            .unwrap()
            .manager_addrs
            .iter()
            .all(|address| *address == hdf5_pure::DecodedStoredAddress::Undefined),
        has_manager
    );
    file.close().unwrap();
    let properties = FileAccessProperties::new().with_memory_strategy(backing);
    let file = File::open_rw_with_options(&path, properties).unwrap();
    let reusable = file.reusable_free_sections().unwrap();
    assert_eq!(reusable.section_count() > 0, has_manager);
    file.root()
        .create_dataset("added", |builder| {
            builder.with_i32_data(&[23; 16]);
        })
        .unwrap();
    file.commit().unwrap();
    let extent = file_space::contiguous_extent(&file, "added");
    if has_manager {
        assert!(
            reusable
                .sections()
                .iter()
                .any(|section| section.address().get() <= extent.start
                    && extent.end <= section.end_address().get())
        );
    }
    file.close().unwrap();
    let file = File::open_rw_with_options(&path, properties).unwrap();
    file.root()
        .create_dataset("third", |builder| {
            builder.with_i32_data(&[29; 16]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        [17; 16]
    );
    assert_eq!(file.dataset("added").unwrap().read_i32().unwrap(), [23; 16]);
    assert_eq!(file.dataset("third").unwrap().read_i32().unwrap(), [29; 16]);
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
}

#[rstest]
fn hdf5_pure_writes_no_free_space_manager_under_aggr_or_none(
    #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::None)] strategy: FileSpaceStrategy,
    #[values(
        aggregation::METADATA_HEAVY,
        aggregation::SMALL_RAW,
        aggregation::ALTERNATING,
        aggregation::GLOBAL_HEAP,
        aggregation::FREED_EXTENT,
        aggregation::BLOCK_EDGES
    )]
    workload: Workload,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workload.h5");

    let file = aggregation::create_file(&path, strategy, true);
    for step in workload.steps {
        step.apply(&file);
        storage::assert_no_managers(&file);
    }
    drop(file);

    let file = File::open(&path).unwrap();
    let info = file.file_space_info().unwrap().unwrap();
    assert_eq!(
        (info.strategy, info.persist, info.manager_addrs.as_slice()),
        (strategy, false, [].as_slice())
    );
    assert_eq!(file.persisted_free_sections().unwrap(), None);
}

#[rstest]
fn none_places_new_payloads_at_or_after_the_previous_eof(
    #[values(
        aggregation::METADATA_HEAVY,
        aggregation::SMALL_RAW,
        aggregation::ALTERNATING,
        aggregation::GLOBAL_HEAP,
        aggregation::FREED_EXTENT,
        aggregation::BLOCK_EDGES
    )]
    workload: Workload,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workload.h5");

    let file = aggregation::create_file(&path, FileSpaceStrategy::None, false);
    let mut strings = Vec::new();
    for &step in workload.steps {
        let before = file.file_size();
        step.apply(&file);
        assert!(file.file_size() >= before);
        if let Step::Create(object) = step {
            match object {
                Object::Dataset(name, _) | Object::Strings(name) => {
                    let placed = file_space::raw_extents(&file, name);
                    assert!(!placed.is_empty());
                    assert!(
                        placed.iter().all(|extent| extent.start >= before),
                        "{step:?}: {placed:?}"
                    );
                }
                Object::Group(name) => {
                    file.group(name).unwrap();
                }
            }
            if let Object::Strings(name) = object {
                strings.push((name, before));
            }
        }
    }

    drop(file);

    let bytes = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    for (name, end) in strings {
        let collection = aggregation::collection_address(
            &bytes,
            file_space::contiguous_extent(&file, name).start,
        );
        assert!(
            collection >= end,
            "{name:?} stores its strings in a collection at {collection}, below the end of the file at {end}"
        );
    }
}

#[rstest]
fn paged_workloads_preserve_survivors_after_reopen(
    #[values(
        aggregation::METADATA_HEAVY,
        aggregation::SMALL_RAW,
        aggregation::ALTERNATING,
        aggregation::GLOBAL_HEAP,
        aggregation::FREED_EXTENT,
        aggregation::BLOCK_EDGES
    )]
    workload: Workload,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workload.h5");

    let file = aggregation::create_file(&path, FileSpaceStrategy::Page, true);
    for step in workload.steps {
        step.apply(&file);
    }
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    for object in workload.surviving_objects() {
        match object {
            Object::Dataset(name, len) => assert_eq!(
                file.dataset(name).unwrap().read_i32().unwrap(),
                vec![aggregation::DATASET_VALUE; len]
            ),
            Object::Strings(name) => assert_eq!(
                file.dataset(name).unwrap().read_string().unwrap(),
                aggregation::STRINGS
            ),
            Object::Group(name) => {
                file.group(name).unwrap();
            }
        }
    }
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
}

#[rstest]
#[case::mirror(MemoryStrategy::Mirrored, 0)]
#[case::bounded(MemoryStrategy::Bounded, 0)]
#[case::userblock(MemoryStrategy::Mirrored, 512)]
fn staged_aggr_allocations_reuse_session_windows(
    #[case] backing: MemoryStrategy,
    #[case] userblock: u64,
    #[values(false, true)] reopen: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("session.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Aggr));
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.write(&path).unwrap();
    let properties = FileAccessProperties::new().with_memory_strategy(backing);
    let mut file = File::open_rw_with_options(&path, properties).unwrap();
    let original_end = file.file_size();
    file.root()
        .create_dataset("first", |builder| {
            builder.with_i32_data(&[23; 16]);
        })
        .unwrap();
    file.commit().unwrap();
    let first = file_space::contiguous_extent(&file, "first");
    let stats = file.reusable_space_stats().unwrap();
    let sections = file.reusable_free_sections().unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        stats.logical_size()
    );
    storage::assert_no_managers(&file);
    assert!(stats.logical_size() >= original_end);
    assert!(sections.section_count() > 0);
    if reopen {
        file.close().unwrap();
        file = File::open_rw_with_options(&path, properties).unwrap();
        assert_eq!(file.reusable_free_sections().unwrap().section_count(), 0);
    }
    let before_second = file.file_size();
    file.root()
        .create_dataset("second", |builder| {
            builder.with_i32_data(&[29; 16]);
        })
        .unwrap();
    file.commit().unwrap();
    let second = file_space::contiguous_extent(&file, "second");
    if reopen {
        assert!(second.start >= before_second);
    } else {
        assert_eq!(second.start, first.end);
        assert_eq!(file.file_size(), before_second);
    }
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("first").unwrap().read_i32().unwrap(), [23; 16]);
    assert_eq!(
        file.dataset("second").unwrap().read_i32().unwrap(),
        [29; 16]
    );
    assert_eq!(file.persisted_free_sections().unwrap(), None);
    assert_eq!(
        file.superblock().eof_address,
        std::fs::metadata(&path).unwrap().len()
    );
}
