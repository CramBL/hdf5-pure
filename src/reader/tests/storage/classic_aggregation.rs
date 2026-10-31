use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;

use rstest::rstest;
use test_util_hdf5::aggregation;
use test_util_hdf5::aggregation::Object;
use test_util_hdf5::aggregation::Step;
use test_util_hdf5::aggregation::Workload;

use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileCreateProperties;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MemoryStrategy;
use crate::PhysicalAddress;
use crate::UserblockSize;
use crate::edit::SpaceOwnership;
use crate::reader::test_util as reader_test_util;

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
    let before = reader_test_util::ownership(&file).unwrap();
    assert!(before.analysis_complete);
    assert_eq!(before.physical_size, before.logical_size);
    assert_eq!(before.trailing_slack_space, []);
    assert_eq!(!before.free_space_manager_space.is_empty(), has_manager);
    if has_manager {
        assert_eq!(before.unclassified_space, []);
    }
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
    let extent = reader_test_util::contiguous_extent(&file, "added");
    if has_manager {
        assert!(
            reusable
                .sections()
                .iter()
                .any(|section| section.address().get() <= extent.start
                    && extent.end <= section.end_address().get())
        );
    }
    assert!(
        reader_test_util::ownership(&file)
            .unwrap()
            .analysis_complete
    );
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
    let after = reader_test_util::ownership(&file).unwrap();
    assert!(after.analysis_complete);
    assert_eq!(after.physical_size, after.logical_size);
    assert_eq!(after.trailing_slack_space, []);
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

    let file = create_file(&path, strategy, true);
    for step in workload.steps {
        apply_step(*step, &file);
        assert_eq!(
            reader_test_util::ownership(&file)
                .unwrap()
                .free_space_manager_space,
            [],
            "{step:?}"
        );
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
fn hdf5_pure_places_every_allocation_of_an_edit_under_none_at_or_past_the_end_of_the_file(
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

    let file = create_file(&path, FileSpaceStrategy::None, false);
    let mut strings = Vec::new();
    for step in workload.steps {
        let before = reader_test_util::ownership(&file).unwrap();
        apply_step(*step, &file);
        let after = reader_test_util::ownership(&file).unwrap();

        let placed = live_extents(&after)
            .difference(&live_extents(&before))
            .copied()
            .collect::<Vec<_>>();
        assert!(
            placed.iter().all(|&(addr, _)| addr >= before.logical_size),
            "{step:?} placed {placed:?} below the end of the file at {}",
            before.logical_size
        );
        if let Step::Create(object) = *step {
            assert!(!placed.is_empty(), "{step:?} placed no extent");
            if let Object::Strings(name) = object {
                strings.push((name, before.logical_size));
            }
        }
    }
    drop(file);

    let bytes = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    for (name, end) in strings {
        let collection = aggregation::collection_address(
            &bytes,
            reader_test_util::contiguous_extent(&file, name).start,
        );
        assert!(
            collection >= end,
            "{name:?} stores its strings in a collection at {collection}, below the end of the file at {end}"
        );
    }
}

#[rstest]
fn hdf5_pure_places_a_workload_under_page_as_recorded(
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

    let file = create_file(&path, FileSpaceStrategy::Page, true);
    let mut report = format!(
        "PAGE allocation workload: {}\n\n\
         All sizes and offsets are in bytes. End offsets are exclusive.\n",
        workload.name
    );
    for (index, &step) in workload.steps.iter().enumerate() {
        apply_step(step, &file);
        write_page_step(
            &mut report,
            index + 1,
            step,
            reader_test_util::ownership(&file).unwrap(),
        );
    }

    insta::assert_snapshot!(format!("page_{}", workload.name), report);
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
    let original_end = reader_test_util::ownership(&file).unwrap().logical_size;
    file.root()
        .create_dataset("first", |builder| {
            builder.with_i32_data(&[23; 16]);
        })
        .unwrap();
    file.commit().unwrap();
    let first = reader_test_util::contiguous_extent(&file, "first");
    let ownership = reader_test_util::ownership(&file).unwrap();
    assert!(ownership.analysis_complete);
    assert_eq!(ownership.physical_size, ownership.logical_size);
    assert_eq!(ownership.free_space_manager_space, []);
    assert_eq!(ownership.trailing_slack_space, []);
    assert!(
        ownership
            .unclassified_space
            .iter()
            .all(|&(start, length)| start + length <= original_end)
    );
    assert!(!ownership.reusable_free_space.is_empty());
    test_util_hdf5::assert_free_sections!(
        &file.reusable_free_sections().unwrap(),
        &ownership.reusable_free_space,
        PhysicalAddress
    );
    if reopen {
        file.close().unwrap();
        file = File::open_rw_with_options(&path, properties).unwrap();
        assert_eq!(file.reusable_free_sections().unwrap().section_count(), 0);
    }
    let before_second = reader_test_util::ownership(&file).unwrap().logical_size;
    file.root()
        .create_dataset("second", |builder| {
            builder.with_i32_data(&[29; 16]);
        })
        .unwrap();
    file.commit().unwrap();
    let second = reader_test_util::contiguous_extent(&file, "second");
    if reopen {
        assert!(second.start >= before_second);
    } else {
        assert_eq!(second.start, first.end);
        assert_eq!(
            reader_test_util::ownership(&file).unwrap().logical_size,
            before_second
        );
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

/// Returns the extents that hold metadata, raw data, or a free-space manager.
fn live_extents(ownership: &SpaceOwnership) -> BTreeSet<(u64, u64)> {
    ownership
        .live_metadata_space
        .iter()
        .chain(&ownership.live_raw_space)
        .chain(&ownership.free_space_manager_space)
        .copied()
        .collect()
}

/// Writes the operation and an extent table sorted by offset within each category.
///
/// # Panics
///
/// Panics if an extent's end offset exceeds [`u64::MAX`].
fn write_page_step(report: &mut String, number: usize, step: Step, ownership: SpaceOwnership) {
    write!(report, "\nStep {number}: ").unwrap();
    match step {
        Step::Create(Object::Dataset(name, len)) => {
            writeln!(report, "Create dataset {name:?} with {len} i32 elements").unwrap();
        }
        Step::Create(Object::Group(name)) => writeln!(report, "Create group {name:?}").unwrap(),
        Step::Create(Object::Strings(name)) => {
            writeln!(report, "Create variable-length string dataset {name:?}").unwrap();
        }
        Step::Delete(name) => writeln!(report, "Delete link {name:?}").unwrap(),
    }
    let SpaceOwnership {
        physical_size,
        logical_size,
        userblock_space,
        live_raw_space,
        live_metadata_space,
        free_space_manager_space,
        reusable_free_space,
        unclassified_space,
        analysis_complete,
        trailing_slack_space,
    } = ownership;
    writeln!(
        report,
        "  Physical size: {physical_size}, logical size: {logical_size}, analysis: {}",
        if analysis_complete {
            "complete"
        } else {
            "incomplete"
        }
    )
    .unwrap();
    writeln!(
        report,
        "  {:<19} {:>10} {:>10} {:>10}",
        "Category", "Start", "End", "Length"
    )
    .unwrap();
    let mut empty = Vec::new();
    for (category, mut extents) in [
        ("Userblock", userblock_space),
        ("Live metadata", live_metadata_space),
        ("Live raw data", live_raw_space),
        ("Free-space managers", free_space_manager_space),
        ("Reusable free space", reusable_free_space),
        ("Unclassified", unclassified_space),
        ("Trailing slack", trailing_slack_space),
    ] {
        if extents.is_empty() {
            empty.push(category);
            continue;
        }
        extents.sort_unstable();
        for (start, length) in extents {
            writeln!(
                report,
                "  {category:<19} {start:>10} {:>10} {length:>10}",
                start.checked_add(length).unwrap()
            )
            .unwrap();
        }
    }
    if !empty.is_empty() {
        writeln!(report, "  No extents: {}", empty.join(", ")).unwrap();
    }
}

fn create_file(path: &Path, strategy: FileSpaceStrategy, persist: bool) -> File {
    File::create_with_options(
        path,
        FileCreateProperties::new()
            .with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(persist)),
        FileAccessProperties::default(),
    )
    .unwrap()
}

fn apply_step(step: Step, file: &File) {
    match step {
        Step::Create(Object::Dataset(name, len)) => {
            file.root()
                .create_dataset(name, |builder| {
                    builder.with_i32_data(&vec![aggregation::DATASET_VALUE; len]);
                })
                .unwrap();
        }
        Step::Create(Object::Group(name)) => {
            file.root().create_group(name).unwrap();
        }
        Step::Create(Object::Strings(name)) => {
            file.root()
                .create_dataset(name, |builder| {
                    builder.with_vlen_strings(aggregation::STRINGS);
                })
                .unwrap();
        }
        Step::Delete(name) => file.root().delete(name).unwrap(),
    }
    file.commit().unwrap();
}
