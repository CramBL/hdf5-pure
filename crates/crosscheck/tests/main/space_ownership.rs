#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5_pure::File;
use rstest::rstest;
use tempfile::TempDir;
use test_util::fractal_heap;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::space_ownership;
use test_util_hdf5::space_ownership::SharedStorage;
use test_util_hdf5::storage;

use crate::heap_allocations;

#[rstest]
#[case::shared_chunked_dataspace(SharedStorage::Dataspace, false)]
#[case::compact_shared_attribute(SharedStorage::Attribute, false)]
#[case::dense_shared_attribute(SharedStorage::Attribute, true)]
#[case::compact_shared_attribute_fields(SharedStorage::AttributeFields, false)]
#[case::dense_shared_attribute_fields(SharedStorage::AttributeFields, true)]
fn shared_metadata_retains_heap_storage_after_owner_deletion(
    #[case] storage: SharedStorage,
    #[case] dense: bool,
) {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("shared.h5");
    space_ownership::libhdf5_write_shared_storage(&path, storage, dense);
    assert_shared_storage_survives_edit(&path, storage, dense);
}

#[test]
fn native_filtered_heap_storage_stays_unavailable_during_an_unrelated_edit() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("filtered.h5");
    space_ownership::libhdf5_write_filtered_heap(&path);
    let bytes = std::fs::read(&path).unwrap();
    let heaps = fractal_heap::header_offsets(&bytes);
    assert_eq!(heaps.len(), 1);
    let allocations = heap_allocations::allocations(&bytes, u64::try_from(heaps[0]).unwrap())
        .into_iter()
        .map(|allocation| allocation.extent())
        .collect::<Vec<_>>();
    let file = File::open_rw(&path).unwrap();
    storage::assert_unavailable(&file, &allocations);
    let mut names = file.group("filtered").unwrap().groups().unwrap();
    names.sort();
    assert_eq!(
        names,
        (0..12)
            .map(|index| format!("g{index:02}"))
            .collect::<Vec<_>>()
    );
    file.root()
        .create_dataset("probe", |builder| {
            builder.with_i32_data(&[19; 32]);
        })
        .unwrap();
    file.commit().unwrap();
    storage::assert_unavailable(&file, &allocations);
    file.close().unwrap();
    let file = hdf5::File::open(&path).unwrap();
    let mut names = file.group("filtered").unwrap().member_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        (0..12)
            .map(|index| format!("g{index:02}"))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        file.dataset("probe").unwrap().read_raw::<i32>().unwrap(),
        [19; 32]
    );
}

fn assert_shared_storage_survives_edit(path: &Path, storage: SharedStorage, dense: bool) {
    let file = File::open_rw(path).unwrap();
    file.root()
        .create_dataset("probe", |builder| {
            builder.with_vlen_strings(&["probe"]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let bytes = std::fs::read(path).unwrap();
    let collections = global_heap::collection_ranges(&bytes, Widths::EIGHT);
    assert_eq!(collections.len(), 1);
    let file = File::open_rw(path).unwrap();
    file.root().delete("probe").unwrap();
    file.commit().unwrap();
    storage::assert_unavailable(
        &file,
        &collections
            .iter()
            .map(|extent| (extent.start, extent.end - extent.start))
            .collect::<Vec<_>>(),
    );
    file.close().unwrap();
    let file = hdf5::File::open(path).unwrap();
    assert!(!file.link_exists("probe"));
    match storage {
        SharedStorage::Dataspace => {
            for index in 0..2 {
                assert_eq!(
                    file.dataset(&format!("d{index}"))
                        .unwrap()
                        .read_raw::<i32>()
                        .unwrap(),
                    [1, 2, 3, 4, 5, 6, 7, 8]
                );
            }
        }
        SharedStorage::Attribute | SharedStorage::AttributeFields => {
            for index in 0..2 {
                let group = file.group(&format!("g{index}")).unwrap();
                if dense {
                    assert_eq!(
                        group.attr("shared").unwrap().read_raw::<i64>().unwrap(),
                        vec![17; 8_192]
                    );
                } else {
                    assert_eq!(
                        group.attr("shared").unwrap().read_scalar::<i64>().unwrap(),
                        17
                    );
                }
            }
        }
    }
}
