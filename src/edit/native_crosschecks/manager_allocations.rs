use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::page_managers::{self, Geometry};

use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::free_space_manager;

#[rstest]
#[case::page(Geometry::Small)]
#[case::userblock(Geometry::Userblock)]
fn native_manager_allocation_visiting_matches_independent_capacity(#[case] geometry: Geometry) {
    let path = test_util::temp::temp_path("native-manager-allocation-visitor.h5");
    page_managers::write_native(&path, geometry);
    let metadata_bytes = file_space_oracle::assert_manager_metadata_bytes(&path);
    assert!(metadata_bytes > 0);
    let original = std::fs::read(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let (_, info) = WriteEngine::extension_fsinfo(&engine.image, &engine.context)
        .unwrap()
        .unwrap();
    let (sections, owned) = free_space_manager::read_persisted_sections_source(
        &engine.image,
        engine.widths(),
        engine.addresses().effective_base(),
        &info.manager_addrs,
    )
    .unwrap();
    assert!(!sections.is_empty());
    let visited = free_space_manager::read_persisted_allocations_source(
        &engine.image,
        engine.widths(),
        engine.addresses().effective_base(),
        &info.manager_addrs,
    )
    .unwrap();
    assert_eq!(visited, owned);
    assert_eq!(
        visited.iter().map(|range| range.length()).sum::<u64>(),
        metadata_bytes
    );
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.member_names().unwrap(),
        ["d0", "d1", "d2", "d3", "d4", "d5"]
    );
    for index in 0_u8..6 {
        assert_eq!(
            native
                .dataset(&format!("d{index}"))
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            vec![index; 1_000]
        );
    }
    native.close().unwrap();
}
