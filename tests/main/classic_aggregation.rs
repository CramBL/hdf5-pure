use std::collections::BTreeSet;
use std::fmt::Write as _;

use hdf5_pure::File;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::SpaceOwnership;
use rstest::rstest;
use tempfile::tempdir;

use test_util_hdf5::aggregation;
use test_util_hdf5::aggregation::Object;
use test_util_hdf5::aggregation::Step;
use test_util_hdf5::aggregation::Workload;
use test_util_hdf5::file_space;

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
    let dir = tempdir().unwrap();
    let path = dir.path().join("workload.h5");

    let file = aggregation::create_file(&path, strategy, true);
    for step in workload.steps {
        step.apply(&file);
        assert_eq!(
            file.__space_ownership().unwrap().free_space_manager_space,
            [],
            "{step:?}"
        );
    }
    drop(file);

    let file = File::open(&path).unwrap();
    let info = file.file_space_info().unwrap();
    assert_eq!(
        (info.strategy, info.persist, info.manager_addrs.as_slice()),
        (strategy, false, [].as_slice())
    );
    assert_eq!(file.persisted_free_space().unwrap(), []);
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
    let dir = tempdir().unwrap();
    let path = dir.path().join("workload.h5");

    let file = aggregation::create_file(&path, FileSpaceStrategy::None, false);
    let mut strings = Vec::new();
    for &step in workload.steps {
        let before = file.__space_ownership().unwrap();
        step.apply(&file);
        let after = file.__space_ownership().unwrap();

        let placed = live_extents(&after)
            .difference(&live_extents(&before))
            .copied()
            .collect::<Vec<_>>();
        assert!(
            placed.iter().all(|&(addr, _)| addr >= before.logical_size),
            "{step:?} placed {placed:?} below the end of the file at {}",
            before.logical_size
        );
        if let Step::Create(object) = step {
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
            file_space::contiguous_extent(&file, name).start,
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
    let dir = tempdir().unwrap();
    let path = dir.path().join("workload.h5");

    let file = aggregation::create_file(&path, FileSpaceStrategy::Page, true);
    let mut report = String::new();
    for step in workload.steps {
        step.apply(&file);
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
        } = file.__space_ownership().unwrap();
        writeln!(
            report,
            "{step:?}: physical size {physical_size}, logical size {logical_size}, \
             userblock {userblock_space:?}, metadata {:?}, raw {:?}, managers {:?}, free {:?}, \
             unclassified {:?}, trailing slack {trailing_slack_space:?}, complete {analysis_complete}",
            sorted(live_metadata_space),
            sorted(live_raw_space),
            sorted(free_space_manager_space),
            sorted(reusable_free_space),
            sorted(unclassified_space),
        )
        .unwrap();
    }

    insta::assert_snapshot!(format!("page_{}", workload.name), report);
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

fn sorted(mut extents: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    extents.sort_unstable();
    extents
}
