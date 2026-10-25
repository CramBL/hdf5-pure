#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5::types::VarLenUnicode;
use hdf5_pure::AttrValue;
use hdf5_pure::DeflateLevel;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FreeSectionCoordinates;
use hdf5_pure::Layout;
use hdf5_pure::MaxExtent;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::UserblockSize;
use rstest::rstest;

use test_util_hdf5::aggregation;
use test_util_hdf5::aggregation::Object;
use test_util_hdf5::aggregation::Step;
use test_util_hdf5::aggregation::Workload;
use test_util_hdf5::file_space;
use test_util_hdf5::storage;

use crate::c_object;

#[rstest]
#[case::tracked(1, true)]
#[case::above_aggregate(4_096, false)]
fn libhdf5_recovers_and_reuses_new_file_aggregate_spare(
    #[case] threshold: u64,
    #[case] has_sections: bool,
    #[values(0, 512)] userblock: u64,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("persisted.h5");
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
    let sections = file.persisted_free_sections().unwrap().unwrap();
    assert_eq!(sections.free_sections().section_count() > 0, has_sections);
    drop(file);
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        let mut expected = file_space::libhdf5_free_sections(&file).unwrap();
        expected.sort_unstable();
        assert_eq!(sections.free_sections().section_count(), expected.len());
        assert_eq!(
            sections.free_sections().free_bytes(),
            expected.iter().map(|&(_, length)| length).sum::<u64>()
        );
        let mut previous_end = 0;
        for (section, (address, length)) in sections.free_sections().sections().iter().zip(expected)
        {
            assert!(length > 0);
            assert!(address >= previous_end);
            previous_end = address.checked_add(length).unwrap();
            assert!(previous_end <= file.size() - userblock);
            assert_eq!(section.coordinates(), FreeSectionCoordinates::Stored);
            assert_eq!(section.address(), address);
            assert_eq!(section.length().get(), length);
        }
        let added = file.new_dataset::<i32>().shape(16).create("added").unwrap();
        added.write_raw(&[23; 16]).unwrap();
        if has_sections {
            let remaining: u64 = file_space::libhdf5_free_sections(&file)
                .unwrap()
                .iter()
                .map(|&(_, size)| size)
                .sum();
            assert!(remaining < sections.free_sections().free_bytes());
        }
    }
    let bytes = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    if has_sections {
        assert_classic_fixture_partition(&file, &bytes, &["values", "added"]);
    }
    let raw = ["values", "added"]
        .iter()
        .flat_map(|name| file_space::raw_extents(&file, name))
        .map(|extent| (extent.start, extent.end - extent.start))
        .collect::<Vec<_>>();
    storage::assert_unavailable(&file, &raw);
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        [17; 16]
    );
    assert_eq!(file.dataset("added").unwrap().read_i32().unwrap(), [23; 16]);
    file.root()
        .create_dataset("third", |builder| {
            builder.with_i32_data(&[29; 16]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    if has_sections {
        assert_classic_fixture_partition(&file, &bytes, &["values", "added", "third"]);
    }
    let raw = ["values", "added", "third"]
        .iter()
        .flat_map(|name| file_space::raw_extents(&file, name))
        .map(|extent| (extent.start, extent.end - extent.start))
        .collect::<Vec<_>>();
    storage::assert_unavailable(&file, &raw);
    file.close().unwrap();
    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(
        file.dataset("third").unwrap().read_raw::<i32>().unwrap(),
        [29; 16]
    );
}

#[rstest]
fn libhdf5_reads_and_edits_an_aggregated_creation(
    #[values(0, 512)] userblock: u64,
    #[values(false, true)] persist: bool,
    #[values(None, Some(FileSpaceStrategy::Aggr), Some(FileSpaceStrategy::FsmAggr))]
    strategy: Option<FileSpaceStrategy>,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("created.h5");
    let mut builder = FileBuilder::new();
    if let Some(strategy) = strategy {
        builder.with_file_space_settings(
            FileSpaceSettings::new(strategy)
                .with_persistence(persist)
                .with_threshold(9),
        );
    }
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.set_attr("huge", AttrValue::AsciiString("h".repeat(70_000)));
    for index in 0..9 {
        builder.set_attr(&format!("attribute_{index}"), AttrValue::I64(index));
    }
    builder.set_attr("label", AttrValue::VarLenString("collection".into()));
    builder.create_dataset("small").with_i32_data(&[11; 17]);
    builder.create_dataset("large").with_u8_data(&[13; 4_097]);
    builder.create_dataset("last").with_u8_data(&[17; 19]);
    builder
        .create_dataset("strings")
        .with_vlen_strings(&["late", "collection"]);
    builder
        .create_dataset("chunked_strings")
        .with_vlen_strings(&["early", "collection"])
        .with_chunks(&[1])
        .with_deflate(DeflateLevel::Six);
    builder
        .create_dataset("fixed")
        .with_i32_data(&[1, 2, 3, 4, 5])
        .with_chunks(&[2]);
    builder
        .create_dataset("extensible")
        .with_i32_data(&[7, 8, 9])
        .with_chunks(&[2])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder.write(&path).unwrap();
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        assert_eq!(
            file.create_plist()
                .unwrap()
                .get_file_space_strategy()
                .unwrap(),
            if strategy == Some(FileSpaceStrategy::Aggr) {
                AGGR
            } else {
                CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: persist && strategy == Some(FileSpaceStrategy::FsmAggr),
                    threshold: if strategy.is_some() { 9 } else { 1 },
                }
            }
        );
        if !(persist && strategy == Some(FileSpaceStrategy::FsmAggr)) {
            assert_eq!(file.free_space(), 0);
        }
        assert_eq!(
            file.dataset("small").unwrap().read_raw::<i32>().unwrap(),
            [11; 17]
        );
        assert_eq!(
            file.dataset("large").unwrap().read_raw::<u8>().unwrap(),
            [13; 4_097]
        );
        assert_eq!(
            file.dataset("last").unwrap().read_raw::<u8>().unwrap(),
            [17; 19]
        );
        assert_eq!(
            file.dataset("fixed").unwrap().read_raw::<i32>().unwrap(),
            [1, 2, 3, 4, 5]
        );
        assert_eq!(
            file.dataset("extensible")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [7, 8, 9]
        );
        assert_eq!(
            file.attr("huge")
                .unwrap()
                .read_scalar::<hdf5::types::FixedAscii<70_000>>()
                .unwrap()
                .as_str(),
            "h".repeat(70_000)
        );
        assert_eq!(
            file.attr("label")
                .unwrap()
                .read_scalar::<VarLenUnicode>()
                .unwrap()
                .as_str(),
            "collection"
        );
        assert_eq!(
            file.dataset("strings")
                .unwrap()
                .read_raw::<VarLenUnicode>()
                .unwrap()
                .iter()
                .map(VarLenUnicode::as_str)
                .collect::<Vec<_>>(),
            ["late", "collection"]
        );
        assert_eq!(
            file.dataset("chunked_strings")
                .unwrap()
                .read_raw::<VarLenUnicode>()
                .unwrap()
                .iter()
                .map(VarLenUnicode::as_str)
                .collect::<Vec<_>>(),
            ["early", "collection"]
        );
        file.dataset("small").unwrap().write_raw(&[23; 17]).unwrap();
        file.new_dataset::<u8>()
            .shape(3)
            .create("added")
            .unwrap()
            .write_raw(&[29, 31, 37])
            .unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("small").unwrap().read_i32().unwrap(), [23; 17]);
    assert_eq!(
        file.dataset("added").unwrap().read_raw().unwrap(),
        [29, 31, 37]
    );
    if !(persist && strategy == Some(FileSpaceStrategy::FsmAggr)) {
        assert_eq!(file.persisted_free_sections().unwrap(), None);
    }
    assert_eq!(
        file.superblock().eof_address,
        std::fs::metadata(path).unwrap().len()
    );
}

#[rstest]
#[case::metadata_heavy_aggr(aggregation::METADATA_HEAVY, AGGR, METADATA_HEAVY_AGGR)]
#[case::metadata_heavy_none(aggregation::METADATA_HEAVY, CStrategy::None, METADATA_HEAVY_NONE)]
#[case::small_raw_aggr(aggregation::SMALL_RAW, AGGR, SMALL_RAW_AGGR)]
#[case::small_raw_none(aggregation::SMALL_RAW, CStrategy::None, SMALL_RAW_NONE)]
#[case::small_raw_page(aggregation::SMALL_RAW, PAGE, SMALL_RAW_PAGE)]
#[case::alternating_aggr(aggregation::ALTERNATING, AGGR, ALTERNATING_AGGR)]
#[case::alternating_none(aggregation::ALTERNATING, CStrategy::None, ALTERNATING_NONE)]
#[case::global_heap_aggr(aggregation::GLOBAL_HEAP, AGGR, GLOBAL_HEAP_AGGR)]
#[case::global_heap_none(aggregation::GLOBAL_HEAP, CStrategy::None, GLOBAL_HEAP_NONE)]
#[case::freed_extent_aggr(aggregation::FREED_EXTENT, AGGR, FREED_EXTENT_AGGR)]
#[case::freed_extent_fsm_aggr(aggregation::FREED_EXTENT, FSM_AGGR, FREED_EXTENT_FSM_AGGR)]
#[case::block_edges_aggr(aggregation::BLOCK_EDGES, AGGR, BLOCK_EDGES_AGGR)]
#[case::block_edges_none(aggregation::BLOCK_EDGES, CStrategy::None, BLOCK_EDGES_NONE)]
fn libhdf5_places_a_workload_as_recorded(
    #[case] workload: Workload,
    #[case] strategy: CStrategy,
    #[case] expected: Placements<&[u64]>,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workload.h5");

    let observed = Placements::observe(&path, workload, strategy);

    assert_eq!(observed.as_slices(), expected);
}

#[test]
fn libhdf5_persists_no_free_space_manager_under_aggr() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("aggr.h5");
    Placements::observe(&path, aggregation::FREED_EXTENT, AGGR);

    let file = File::open(&path).unwrap();
    let info = file.file_space_info().unwrap().unwrap();
    assert_eq!(
        (info.strategy, info.persist, info.manager_addrs.as_slice()),
        (FileSpaceStrategy::Aggr, false, [].as_slice())
    );
    assert_eq!(file.persisted_free_sections().unwrap(), None);
    let reopened = hdf5::File::open(&path).unwrap();
    assert_eq!(reopened.free_space(), 0);
    assert_eq!(file_space::libhdf5_free_sections(&reopened).unwrap(), []);
}

#[rstest]
fn libhdf5_reads_the_workload_hdf5_pure_writes(
    #[values(
        aggregation::METADATA_HEAVY,
        aggregation::SMALL_RAW,
        aggregation::ALTERNATING,
        aggregation::GLOBAL_HEAP,
        aggregation::FREED_EXTENT,
        aggregation::BLOCK_EDGES
    )]
    workload: Workload,
    #[values(
        FileSpaceStrategy::Aggr,
        FileSpaceStrategy::None,
        FileSpaceStrategy::FsmAggr,
        FileSpaceStrategy::Page
    )]
    strategy: FileSpaceStrategy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workload.h5");
    workload.write(&path, strategy, true);

    let file = hdf5::File::open(&path).unwrap();
    let mut members = file.member_names().unwrap();
    members.sort_unstable();
    let mut expected = workload
        .surviving_objects()
        .map(Object::link_name)
        .collect::<Vec<_>>();
    expected.sort_unstable();
    assert_eq!(members, expected);
    for object in workload.surviving_objects() {
        match object {
            Object::Dataset(name, len) => assert_eq!(
                file.dataset(name).unwrap().read_raw::<i32>().unwrap(),
                vec![aggregation::DATASET_VALUE; len]
            ),
            Object::Group(name) => {
                file.group(name).unwrap();
            }
            Object::Strings(name) => assert_eq!(
                file.dataset(name)
                    .unwrap()
                    .read_raw::<VarLenUnicode>()
                    .unwrap()
                    .iter()
                    .map(VarLenUnicode::as_str)
                    .collect::<Vec<_>>(),
                aggregation::STRINGS
            ),
        }
    }
}

#[rstest]
#[case::metadata_heavy(aggregation::METADATA_HEAVY)]
#[case::small_raw(aggregation::SMALL_RAW)]
#[case::alternating(aggregation::ALTERNATING)]
#[case::global_heap(aggregation::GLOBAL_HEAP)]
#[case::freed_extent(aggregation::FREED_EXTENT)]
#[case::block_edges(aggregation::BLOCK_EDGES)]
fn libhdf5_reads_and_mutates_its_classic_file_after_staged_edits(
    #[case] workload: Workload,
    #[values((MemoryStrategy::Mirrored, 0), (MemoryStrategy::Bounded, 0), (MemoryStrategy::Mirrored, 512))]
    (backing, userblock): (MemoryStrategy, u64),
    #[values(AGGR, FSM_AGGR, PERSISTENT_FSM_AGGR)] strategy: CStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("staged.h5");
    let seed = hdf5::FileBuilder::new()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| {
            properties
                .file_space_strategy(strategy)
                .userblock(userblock)
        })
        .create(&path)
        .unwrap();
    seed.new_dataset::<u8>()
        .shape(1)
        .create("keep")
        .unwrap()
        .write(&[17])
        .unwrap();
    seed.close().unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(backing),
    )
    .unwrap();
    for step in workload.steps {
        step.apply(&file);
        if strategy != PERSISTENT_FSM_AGGR {
            storage::assert_no_managers(&file);
            let error = file.persisted_free_sections().unwrap_err();
            assert!(
                matches!(error, Error::FreeSectionInspectionUnsupported),
                "{error:?}"
            );
        }
    }
    file.close().unwrap();
    let file = hdf5::File::open_rw(&path).unwrap();
    assert_eq!(
        file.dataset("keep").unwrap().read_raw::<u8>().unwrap(),
        [17]
    );
    for object in workload.surviving_objects() {
        match object {
            Object::Dataset(name, len) => assert_eq!(
                file.dataset(name).unwrap().read_raw::<i32>().unwrap(),
                vec![aggregation::DATASET_VALUE; len],
            ),
            Object::Group(name) => {
                file.group(name).unwrap();
            }
            Object::Strings(name) => assert_eq!(
                file.dataset(name)
                    .unwrap()
                    .read_raw::<VarLenUnicode>()
                    .unwrap()
                    .iter()
                    .map(VarLenUnicode::as_str)
                    .collect::<Vec<_>>(),
                aggregation::STRINGS,
            ),
        }
    }
    if strategy != PERSISTENT_FSM_AGGR {
        assert_eq!(file.free_space(), 0);
    }
    file.new_dataset::<u8>()
        .shape(1)
        .create("c_added")
        .unwrap()
        .write(&[31])
        .unwrap();
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("c_added").unwrap().read_u8().unwrap(), [31]);
}

#[rstest]
#[case::small(6, 4, 14)]
#[case::raw_refill(6, 4, 6000)]
#[case::index_blocks(4, 1, 1500)]
fn libhdf5_reads_and_extends_its_classic_file_after_immediate_appends(
    #[case] initial: usize,
    #[case] chunk: usize,
    #[case] end: usize,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(false, true)] filtered: bool,
    #[values(AGGR, FSM_AGGR, PERSISTENT_FSM_AGGR)] strategy: CStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("immediate.h5");
    let seed = hdf5::FileBuilder::new()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| properties.file_space_strategy(strategy))
        .create(&path)
        .unwrap();
    let mut builder = seed.new_dataset::<u8>();
    if filtered {
        builder = builder.deflate(6);
    }
    let dataset = builder
        .shape((initial..,))
        .chunk([chunk])
        .create("values")
        .unwrap();
    let expected = (0..end)
        .map(|value| (value % 251) as u8)
        .collect::<Vec<_>>();
    dataset.write(&expected[..initial]).unwrap();
    drop(dataset);
    seed.close().unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(backing),
    )
    .unwrap();
    file.root()
        .create_dataset("first", |dataset| {
            dataset.with_u8_data(&[17; 16]);
        })
        .unwrap();
    file.commit().unwrap();
    let before_raw = ["first", "values"]
        .iter()
        .flat_map(|name| file_space::raw_extents(&file, name))
        .collect::<Vec<_>>();
    let raw_cursor = before_raw.iter().map(|extent| extent.end).max().unwrap();
    let reusable = file.reusable_free_sections().unwrap();
    file.dataset("values")
        .unwrap()
        .append_raw(&expected[initial..])
        .unwrap();
    let raw = ["first", "values"]
        .iter()
        .flat_map(|name| file_space::raw_extents(&file, name))
        .collect::<Vec<_>>();
    storage::assert_unavailable(
        &file,
        &raw.iter()
            .map(|extent| (extent.start, extent.end - extent.start))
            .collect::<Vec<_>>(),
    );
    if strategy != PERSISTENT_FSM_AGGR {
        storage::assert_no_managers(&file);
    }
    let placed = raw
        .iter()
        .filter(|extent| !before_raw.contains(extent))
        .collect::<Vec<_>>();
    if strategy == AGGR {
        assert!(
            placed.iter().any(|extent| extent.start == raw_cursor),
            "cursor={raw_cursor}, placed={placed:?}"
        );
    } else {
        assert!(placed.iter().any(|extent| {
            reusable.sections().iter().any(|section| {
                section.address() < extent.end && extent.start < section.end_address()
            })
        }));
    }
    file.root()
        .create_dataset("last", |dataset| {
            dataset.with_u8_data(&[19; 16]);
        })
        .unwrap();
    file.commit().unwrap();
    file.dataset("values")
        .unwrap()
        .append_raw(&[97, 98])
        .unwrap();
    file.close().unwrap();
    let file = hdf5::File::open_rw(&path).unwrap();
    assert_eq!(
        file.dataset("first").unwrap().read_raw::<u8>().unwrap(),
        [17; 16]
    );
    assert_eq!(
        file.dataset("last").unwrap().read_raw::<u8>().unwrap(),
        [19; 16]
    );
    let mut expected = expected;
    expected.extend_from_slice(&[97, 98]);
    if strategy != PERSISTENT_FSM_AGGR {
        assert_eq!(file.free_space(), 0);
    }
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.read_raw::<u8>().unwrap(), expected);
    dataset.resize([end + 4]).unwrap();
    dataset.write_slice(&[101, 102], end + 2..end + 4).unwrap();
    drop(dataset);
    file.close().unwrap();
    expected.extend_from_slice(&[101, 102]);
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("values")
            .unwrap()
            .read_u8()
            .unwrap(),
        expected
    );
}

#[rstest]
fn libhdf5_fsm_file_reuses_deleted_storage_before_aggregate_spare(
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(FSM_AGGR, PERSISTENT_FSM_AGGR)] strategy: CStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fsm_reuse.h5");
    let seed = hdf5::FileBuilder::new()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| properties.file_space_strategy(strategy))
        .create(&path)
        .unwrap();
    seed.new_dataset::<u8>()
        .shape((4..,))
        .chunk([4])
        .create("values")
        .unwrap()
        .write(&[1, 2, 3, 4])
        .unwrap();
    let scratch = seed
        .new_dataset::<u8>()
        .shape(4096)
        .create("scratch")
        .unwrap();
    scratch.write(&[7; 4096]).unwrap();
    let hole = scratch.offset().unwrap();
    drop(scratch);
    seed.new_dataset::<u8>()
        .shape(4096)
        .create("keep")
        .unwrap()
        .write(&[11; 4096])
        .unwrap();
    seed.link_hard("keep", "alias").unwrap();
    seed.close().unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(backing),
    )
    .unwrap();
    file.root()
        .create_dataset("first", |dataset| {
            dataset.with_u8_data(&[17; 64]);
        })
        .unwrap();
    file.commit().unwrap();
    file.root().delete("scratch").unwrap();
    file.root().delete("keep").unwrap();
    file.commit().unwrap();
    let before = file.reusable_free_sections().unwrap();
    let free = before
        .sections()
        .iter()
        .find(|section| section.address() <= hole && hole + 4096 <= section.end_address())
        .unwrap();
    assert_eq!(free.coordinates(), FreeSectionCoordinates::Absolute);
    let free_start = free.address();
    let free_size = free.length().get();
    file.root()
        .create_dataset("replacement", |dataset| {
            dataset.with_u8_data(&[19; 3072]);
        })
        .unwrap();
    file.commit().unwrap();
    let Layout::Contiguous {
        address: Some(address),
        size,
    } = file.dataset("replacement").unwrap().layout().unwrap()
    else {
        panic!("expected contiguous replacement");
    };
    assert_eq!(size, 3072);
    assert!(free_start <= address && address + size <= free_start + free_size);
    file.dataset("values")
        .unwrap()
        .append_raw(&[5; 1024])
        .unwrap();
    let chunks = file.dataset("values").unwrap().chunks().unwrap();
    assert!(chunks.iter().any(|chunk| free_start <= chunk.address
        && chunk.address + chunk.storage_size <= free_start + free_size));
    if strategy != PERSISTENT_FSM_AGGR {
        storage::assert_no_managers(&file);
    }
    let raw = ["replacement", "alias", "values"]
        .iter()
        .flat_map(|name| file_space::raw_extents(&file, name))
        .map(|extent| (extent.start, extent.end - extent.start))
        .collect::<Vec<_>>();
    storage::assert_unavailable(&file, &raw);
    file.close().unwrap();
    let file = hdf5::File::open_rw(&path).unwrap();
    assert_eq!(
        file.dataset("replacement")
            .unwrap()
            .read_raw::<u8>()
            .unwrap(),
        [19; 3072]
    );
    assert_eq!(
        file.dataset("alias").unwrap().read_raw::<u8>().unwrap(),
        [11; 4096]
    );
    let mut expected = vec![1, 2, 3, 4];
    expected.extend_from_slice(&[5; 1024]);
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.read_raw::<u8>().unwrap(), expected);
    dataset.resize([1032]).unwrap();
    dataset.write_slice(&[6; 4], 1028..1032).unwrap();
    drop(dataset);
    file.close().unwrap();
    expected.extend_from_slice(&[6; 4]);
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("values")
            .unwrap()
            .read_u8()
            .unwrap(),
        expected
    );
}

/// Where the C library places the allocations of a workload, and the file size after each step.
#[derive(Debug, Default, Eq, PartialEq)]
struct Placements<T> {
    /// The object header address of each object the workload creates, in step order.
    headers: T,
    /// The raw data address of each dataset the workload creates, in step order.
    raw: T,
    /// The address of the global heap collection of each dataset of strings, in step order.
    collections: T,
    /// The file size `H5Fget_filesize` returns after each step, the larger of the end of
    /// allocation and the end of file.
    file_sizes: T,
    /// The bytes left in the aggregators after each step.
    aggregator_bytes: T,
    /// The length of the file after the C library closes it.
    closed_len: u64,
}

impl Placements<Vec<u64>> {
    /// Runs `workload` through the C library on a new file at `path` under `strategy`, with 1.10 as
    /// the low bound on the library version.
    fn observe(path: &Path, workload: Workload, strategy: CStrategy) -> Self {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| fcpl.file_space_strategy(strategy))
            .create(path)
            .unwrap();
        let mut placements = Self::default();
        let mut strings = Vec::new();
        for &step in workload.steps {
            match step {
                Step::Create(object) => {
                    let raw = create(&file, object).map(|dataset| dataset.offset().unwrap());
                    placements
                        .headers
                        .push(c_object::header_address(&file, object.link_name()));
                    placements.raw.extend(raw);
                    if let Object::Strings(_) = object {
                        strings.extend(raw);
                    }
                }
                Step::Delete(name) => file.unlink(name).unwrap(),
            }
            placements.file_sizes.push(file.size());
            placements.aggregator_bytes.push(aggregator_bytes(&file));
        }
        file.close().unwrap();

        let bytes = std::fs::read(path).unwrap();
        placements.collections = strings
            .into_iter()
            .map(|offset| aggregation::collection_address(&bytes, offset))
            .collect();
        placements.closed_len = std::fs::metadata(path).unwrap().len();
        placements
    }

    fn as_slices(&self) -> Placements<&[u64]> {
        let Self {
            headers,
            raw,
            collections,
            file_sizes,
            aggregator_bytes,
            closed_len,
        } = self;
        Placements {
            headers,
            raw,
            collections,
            file_sizes,
            aggregator_bytes,
            closed_len: *closed_len,
        }
    }
}

/// Creates `object` in the root group of `file`, and returns the dataset if `object` is one.
fn create(file: &hdf5::File, object: Object) -> Option<hdf5::Dataset> {
    match object {
        Object::Dataset(name, len) => {
            let dataset = file.new_dataset::<i32>().shape(len).create(name).unwrap();
            dataset
                .write(&vec![aggregation::DATASET_VALUE; len])
                .unwrap();
            Some(dataset)
        }
        Object::Group(name) => {
            file.create_group(name).unwrap();
            None
        }
        Object::Strings(name) => {
            let values = aggregation::STRINGS
                .iter()
                .map(|value| value.parse::<VarLenUnicode>().unwrap())
                .collect::<Vec<_>>();
            let dataset = file
                .new_dataset::<VarLenUnicode>()
                .shape(values.len())
                .create(name)
                .unwrap();
            dataset.write(&values).unwrap();
            Some(dataset)
        }
    }
}

/// Returns the bytes left in the metadata and small-data aggregators of `file`.
///
/// `H5Fget_freespace` counts these bytes together with the sections the free-space managers track
/// (`H5MF_get_freespace` in `H5MF.c`, HDF5 2.2.0).
fn aggregator_bytes(file: &hdf5::File) -> u64 {
    let tracked: u64 = file_space::libhdf5_free_sections(file)
        .unwrap()
        .iter()
        .map(|&(_, size)| size)
        .sum();
    file.free_space() - tracked
}

/// Asserts a byte partition from serialized headers, dataset payloads, and reusable space.
///
/// # Panics
///
/// Panics if a fixture structure cannot be read, the ranges fail to partition the byte image, or
/// the backing-store length differs from the image length.
fn assert_classic_fixture_partition(file: &File, bytes: &[u8], datasets: &[&str]) {
    let mut extents = storage::file_metadata_extents(file, bytes);
    for name in datasets {
        let address = storage::root_object_address(file, bytes, name);
        extents.extend(storage::header_extents(file, bytes, address));
        extents.extend(
            file_space::raw_extents(file, name)
                .iter()
                .map(|raw| (raw.start, raw.end - raw.start)),
        );
    }
    extents.extend(
        file.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .map(|section| {
                assert_eq!(section.coordinates(), FreeSectionCoordinates::Absolute);
                (section.address(), section.length().get())
            }),
    );
    let userblock = file.superblock().base_address.get();
    if userblock > 0 {
        extents.push((0, userblock));
    }
    extents.sort_unstable();
    let mut end = 0;
    for &(start, len) in &extents {
        assert!(len > 0, "empty fixture extent at {start}");
        assert_eq!(
            start, end,
            "fixture partition has a gap or overlap: {extents:?}"
        );
        end = start.checked_add(len).unwrap();
    }
    let physical = u64::try_from(bytes.len()).unwrap();
    assert_eq!(end, physical, "fixture partition must cover every byte");
    assert_eq!(file.file_size(), physical);
}

/// `H5F_FSPACE_STRATEGY_AGGR`, which hdf5-metno names `PageAggregation`.
const AGGR: CStrategy = CStrategy::PageAggregation;
const FSM_AGGR: CStrategy = CStrategy::FreeSpaceManager {
    paged: false,
    persist: false,
    threshold: 1,
};

const PERSISTENT_FSM_AGGR: CStrategy = CStrategy::FreeSpaceManager {
    paged: false,
    persist: true,
    threshold: 1,
};
const PAGE: CStrategy = CStrategy::FreeSpaceManager {
    paged: true,
    persist: false,
    threshold: 1,
};

const METADATA_HEAVY_AGGR: Placements<&[u64]> = Placements {
    headers: &[255, 402, 549, 696, 843, 990],
    raw: &[],
    collections: &[],
    file_sizes: &[2048, 2048, 2048, 2048, 2048, 2048],
    aggregator_bytes: &[1646, 1499, 1352, 1205, 1058, 864],
    closed_len: 1184,
};

const METADATA_HEAVY_NONE: Placements<&[u64]> = Placements {
    headers: &[255, 402, 549, 696, 843, 990],
    raw: &[],
    collections: &[],
    file_sizes: &[402, 549, 696, 843, 990, 1184],
    aggregator_bytes: &[0, 0, 0, 0, 0, 0],
    closed_len: 1184,
};

const SMALL_RAW_AGGR: Placements<&[u64]> = Placements {
    headers: &[255, 523, 791, 1059, 1327, 1595],
    raw: &[2048, 2112, 2176, 2240, 2304, 2368],
    collections: &[],
    file_sizes: &[4096, 4096, 4096, 4096, 4096, 4096],
    aggregator_bytes: &[3509, 3177, 2845, 2513, 2181, 1802],
    closed_len: 2432,
};

const SMALL_RAW_NONE: Placements<&[u64]> = Placements {
    headers: &[255, 587, 919, 1251, 1583, 1915],
    raw: &[523, 855, 1187, 1519, 1851, 2230],
    collections: &[],
    file_sizes: &[587, 919, 1251, 1583, 1915, 2294],
    aggregator_bytes: &[0, 0, 0, 0, 0, 0],
    closed_len: 2294,
};

const SMALL_RAW_PAGE: Placements<&[u64]> = Placements {
    headers: &[255, 523, 791, 1059, 1327, 1595],
    raw: &[4096, 4160, 4224, 4288, 4352, 4416],
    collections: &[],
    file_sizes: &[8192, 8192, 8192, 8192, 8192, 8192],
    aggregator_bytes: &[0, 0, 0, 0, 0, 0],
    closed_len: 8192,
};

const ALTERNATING_AGGR: Placements<&[u64]> = Placements {
    headers: &[255, 402, 670, 817, 1085, 1232],
    raw: &[2048, 2112, 2176],
    collections: &[],
    file_sizes: &[2048, 4096, 4096, 4096, 4096, 4096],
    aggregator_bytes: &[1646, 3362, 3215, 2883, 2736, 2357],
    closed_len: 2240,
};

const ALTERNATING_NONE: Placements<&[u64]> = Placements {
    headers: &[255, 402, 734, 881, 1213, 1360],
    raw: &[670, 1149, 1675],
    collections: &[],
    file_sizes: &[402, 734, 881, 1213, 1360, 1739],
    aggregator_bytes: &[0, 0, 0, 0, 0, 0],
    closed_len: 1739,
};

const GLOBAL_HEAP_AGGR: Placements<&[u64]> = Placements {
    headers: &[255, 523, 791, 1059],
    raw: &[2048, 2112, 6240],
    collections: &[2144],
    file_sizes: &[4096, 8192, 8192, 8192],
    aggregator_bytes: &[3509, 3209, 2877, 2730],
    closed_len: 6304,
};

const GLOBAL_HEAP_NONE: Placements<&[u64]> = Placements {
    headers: &[255, 587, 4983, 5315],
    raw: &[523, 855, 5251],
    collections: &[887],
    file_sizes: &[587, 4983, 5315, 5462],
    aggregator_bytes: &[0, 0, 0, 0],
    closed_len: 5462,
};

const FREED_EXTENT_AGGR: Placements<&[u64]> = Placements {
    headers: &[255, 523, 791],
    raw: &[2048, 3648, 3712],
    collections: &[],
    file_sizes: &[4096, 4096, 4096, 4096],
    aggregator_bytes: &[1973, 1641, 1641, 1309],
    closed_len: 3776,
};

const FREED_EXTENT_FSM_AGGR: Placements<&[u64]> = Placements {
    headers: &[195, 463, 195],
    raw: &[2048, 3648, 2048],
    collections: &[],
    file_sizes: &[4096, 4096, 4096, 4096],
    aggregator_bytes: &[2033, 1701, 1701, 1701],
    closed_len: 3712,
};

const BLOCK_EDGES_AGGR: Placements<&[u64]> = Placements {
    headers: &[255, 523, 791, 1059, 1327, 1595, 7376, 7691],
    raw: &[2048, 2112, 6112, 6176, 6576, 6976, 9424, 9824],
    collections: &[],
    file_sizes: &[4096, 8096, 8096, 8096, 8096, 8096, 11472, 11472],
    aggregator_bytes: &[3509, 3241, 2909, 2241, 1573, 858, 3381, 2666],
    closed_len: 10224,
};

const BLOCK_EDGES_NONE: Placements<&[u64]> = Placements {
    headers: &[255, 587, 4855, 5187, 5855, 6523, 7238, 7953],
    raw: &[523, 855, 5123, 5455, 6123, 6838, 7553, 8268],
    collections: &[],
    file_sizes: &[587, 4855, 5187, 5855, 6523, 7238, 7953, 8668],
    aggregator_bytes: &[0, 0, 0, 0, 0, 0, 0, 0],
    closed_len: 8668,
};
