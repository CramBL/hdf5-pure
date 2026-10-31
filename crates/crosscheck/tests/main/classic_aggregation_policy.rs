#![cfg(feature = "__hdf5-1.10")]

use core::num::NonZeroU64;

use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5::types::VarLenUnicode;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_format::__private::LengthWidth;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::ClassicAggregators;
use hdf5_pure_space::__private::Extent;
use rstest::rstest;

use test_util_hdf5::aggregation;
use test_util_hdf5::aggregation::Object;
use test_util_hdf5::aggregation::Step;
use test_util_hdf5::aggregation::Workload;

#[rstest]
#[case::metadata_heavy(aggregation::METADATA_HEAVY)]
#[case::small_raw(aggregation::SMALL_RAW)]
#[case::alternating(aggregation::ALTERNATING)]
#[case::global_heap(aggregation::GLOBAL_HEAP)]
#[case::freed_extent(aggregation::FREED_EXTENT)]
#[case::block_edges(aggregation::BLOCK_EDGES)]
fn raw_policy_matches_libhdf5_workload_allocations(#[case] workload: Workload) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("aggregation.h5");
    // Creating metadata first and H5Pset_meta_block_size(0) isolate raw aggregation.
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_v110().meta_block_size(0))
        .with_fcpl(|fcpl| fcpl.file_space_strategy(CStrategy::PageAggregation))
        .create(&path)
        .unwrap();
    for step in workload.steps {
        match step {
            Step::Create(Object::Dataset(name, len)) => {
                file.new_dataset::<i32>().shape(*len).create(*name).unwrap();
            }
            Step::Create(Object::Group(name)) => {
                file.create_group(name).unwrap();
            }
            Step::Create(Object::Strings(name)) => {
                file.new_dataset::<VarLenUnicode>()
                    .shape(aggregation::STRINGS.len())
                    .create(*name)
                    .unwrap();
            }
            Step::Delete(_) => {}
        }
    }
    let mut eoa = file.size();
    let mut observations = Vec::new();
    for &step in workload.steps {
        let allocation = match step {
            Step::Create(Object::Dataset(name, len)) => {
                let dataset = file.dataset(name).unwrap();
                dataset
                    .write(&vec![aggregation::DATASET_VALUE; len])
                    .unwrap();
                Some(Extent::new(dataset.offset().unwrap(), dataset.storage_size()).unwrap())
            }
            Step::Create(Object::Strings(name)) => {
                let dataset = file.dataset(name).unwrap();
                dataset
                    .write(
                        &aggregation::STRINGS
                            .iter()
                            .map(|value| value.parse::<VarLenUnicode>().unwrap())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                Some(Extent::new(dataset.offset().unwrap(), dataset.storage_size()).unwrap())
            }
            Step::Create(Object::Group(_)) => None,
            Step::Delete(name) => {
                file.unlink(name).unwrap();
                None
            }
        };
        observations.push((step, allocation, file.size(), file.free_space()));
    }
    file.close().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), bytes.len() as u64).unwrap(),
        &bytes,
    )
    .unwrap();
    let mut aggregates = ClassicAggregators::new();
    for (step, allocation, end, spare) in observations {
        if let Some(expected) = allocation {
            let allocation = aggregates
                .allocate(
                    eoa,
                    NonZeroU64::new(expected.len()).unwrap(),
                    AllocationClass::Raw,
                )
                .unwrap();
            assert_eq!(allocation.extent(), expected, "{step:?}");
            assert_eq!(allocation.released(), None);
            eoa = allocation.end_of_allocation();
            if let Step::Create(Object::Strings(_)) = step {
                let address = aggregation::collection_address(&bytes, expected.start());
                let locator = GlobalHeapCollectionLocator::new(
                    PhysicalAddress::new(address),
                    LengthWidth::Eight,
                );
                let collection = locator
                    .parse(region.subregion(locator.parse_range().unwrap()).unwrap())
                    .unwrap();
                let allocation = aggregates
                    .allocate(
                        eoa,
                        NonZeroU64::new(collection.range().length()).unwrap(),
                        AllocationClass::Raw,
                    )
                    .unwrap();
                assert_eq!(allocation.extent().start(), address);
                assert_eq!(allocation.released(), None);
                eoa = allocation.end_of_allocation();
            }
        }
        assert_eq!(eoa, end, "{step:?}");
        assert_eq!(
            aggregates
                .unused(AllocationClass::Raw)
                .map_or(0, Extent::len),
            spare,
            "{step:?}"
        );
        assert_eq!(aggregates.unused(AllocationClass::Metadata), None);
    }
    let released = aggregates.reset(eoa).unwrap();
    assert_eq!(
        released.end_of_allocation(),
        std::fs::metadata(&path).unwrap().len()
    );
    assert_eq!(released.released(AllocationClass::Raw), None);
    assert_eq!(released.released(AllocationClass::Metadata), None);

    let reopened = hdf5::File::open(&path).unwrap();
    assert_eq!(reopened.free_space(), 0);
    for object in workload.surviving_objects() {
        match object {
            Object::Dataset(name, len) => assert_eq!(
                reopened.dataset(name).unwrap().read_raw::<i32>().unwrap(),
                vec![aggregation::DATASET_VALUE; len]
            ),
            Object::Group(name) => {
                reopened.group(name).unwrap();
            }
            Object::Strings(name) => assert_eq!(
                reopened
                    .dataset(name)
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
