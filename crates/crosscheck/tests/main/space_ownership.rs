#![cfg(feature = "__hdf5-1.10")]

use hdf5_pure::File;
use rstest::rstest;
use tempfile::TempDir;
use test_util::fractal_heap;
use test_util::range;
use test_util_hdf5::space_ownership;
use test_util_hdf5::space_ownership::SharedStorage;

#[rstest]
#[case::shared_chunked_dataspace(SharedStorage::Dataspace, false)]
#[case::compact_shared_attribute(SharedStorage::Attribute, false)]
#[case::dense_shared_attribute(SharedStorage::Attribute, true)]
#[case::compact_shared_attribute_fields(SharedStorage::AttributeFields, false)]
#[case::dense_shared_attribute_fields(SharedStorage::AttributeFields, true)]
fn native_shared_storage_has_incomplete_ownership(
    #[case] storage: SharedStorage,
    #[case] dense: bool,
) {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("shared.h5");
    space_ownership::libhdf5_write_shared_storage(&path, storage, dense);
    let file = File::open_rw(&path).unwrap();
    let ownership = file.__space_ownership().unwrap();
    assert!(!ownership.analysis_complete);
    assert!(!ownership.unclassified_space.is_empty());
    file.close().unwrap();
}

#[test]
fn a_native_filtered_heap_has_complete_ownership() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("filtered.h5");
    space_ownership::libhdf5_write_filtered_heap(&path);
    let bytes = std::fs::read(&path).unwrap();
    let heaps = fractal_heap::header_offsets(&bytes);
    assert_eq!(heaps.len(), 1);
    let allocations =
        hdf5_pure::__fractal_heap_storage_extents(&bytes, u64::try_from(heaps[0]).unwrap(), 8, 8)
            .unwrap();
    let file = File::open_rw(&path).unwrap();
    let ownership = file.__space_ownership().unwrap();
    assert!(ownership.analysis_complete);
    for (address, len) in allocations {
        assert_eq!(
            range::covered_len(&ownership.live_metadata_space, &(address..address + len)),
            len,
            "the heap allocation ({address}, {len}) is live metadata"
        );
    }
    file.close().unwrap();
}
