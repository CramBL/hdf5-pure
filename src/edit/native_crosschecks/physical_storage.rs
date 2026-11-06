use hdf5_pure_space::__private::{FreeClass, RetainedKind};
use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::page_managers;

use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::physical::claims::ClaimOrigin;
use crate::edit::analysis::physical::{CapacityKind, FileStorageKind};
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;

#[rstest]
#[case::legacy(false)]
#[case::paged_userblock(true)]
fn native_file_storage_matches_framing_and_owned_manager_capacity(#[case] paged: bool) {
    let path = test_util::temp::temp_path("native-physical-storage.h5");
    let metadata_bytes = if paged {
        page_managers::write_native(&path, page_managers::Geometry::Userblock);
        let bytes = file_space_oracle::assert_manager_metadata_bytes(&path);
        assert!(bytes > 0);
        bytes
    } else {
        let native = file::libhdf5_create_earliest(&path);
        native
            .new_dataset::<i32>()
            .shape([3])
            .create("data")
            .unwrap()
            .write(&[7, 11, 13])
            .unwrap();
        native.close().unwrap();
        0
    };
    let native_sections = if paged {
        let checkpoint = file_space_oracle::observe(&path, "capacity", "native-closed");
        assert!(checkpoint.c.bytes() > 0);
        checkpoint.c.sections().to_vec()
    } else {
        vec![]
    };
    let original = std::fs::read(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let object =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, object).unwrap();
    let facts = phase.physical_storage(&source).unwrap();
    assert!(facts.problems().is_empty(), "{:?}", facts.problems());
    assert_eq!(
        facts.physical_size(),
        u64::try_from(original.len()).unwrap()
    );
    assert_eq!(facts.logical_eoa(), u64::try_from(original.len()).unwrap());
    assert_eq!(
        facts
            .file()
            .iter()
            .filter(|entry| entry.kind == FileStorageKind::Manager)
            .map(|entry| entry.range.length())
            .sum::<u64>(),
        metadata_bytes
    );
    let mut actual = facts
        .non_live()
        .iter()
        .map(|entry| {
            (
                engine
                    .addresses()
                    .relative(entry.range().start())
                    .unwrap()
                    .get(),
                entry.range().length(),
            )
        })
        .collect::<Vec<_>>();
    actual.sort_unstable();
    assert_eq!(actual, native_sections);
    assert!(facts.non_live().iter().all(|entry| matches!(
        entry.kind(),
        CapacityKind::Reusable(FreeClass::Page(_))
            | CapacityKind::Retained(RetainedKind::Unclassified)
    )));
    let base = if paged { 8_192 } else { 0 };
    assert_eq!(engine.context.signature().get(), base);
    assert_eq!(
        facts
            .file()
            .iter()
            .filter(|entry| entry.kind == FileStorageKind::Userblock)
            .map(|entry| (entry.range.start().get(), entry.range.length()))
            .collect::<Vec<_>>(),
        if paged { vec![(0, base)] } else { vec![] }
    );
    if paged {
        for manager in facts
            .file()
            .iter()
            .filter(|entry| entry.kind == FileStorageKind::Manager)
        {
            assert!(
                facts
                    .metadata_capacity()
                    .iter()
                    .any(|capacity| capacity.contains(manager.range))
            );
        }
    } else {
        assert!(facts.metadata_capacity().is_empty());
    }
    let expected_metadata = facts
        .metadata_capacity()
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    let expected_capacity = facts
        .non_live()
        .iter()
        .enumerate()
        .map(|(index, entry)| (index, entry.range(), entry.kind()))
        .collect::<Vec<_>>();
    let claims = phase.physical_claims(&source).unwrap();
    assert_eq!(
        claims
            .claims()
            .iter()
            .filter_map(|claim| match claim.origin() {
                ClaimOrigin::MetadataCapacity(index) => Some((index, claim.range())),
                _ => None,
            })
            .collect::<Vec<_>>(),
        expected_metadata
    );
    assert_eq!(
        claims
            .claims()
            .iter()
            .filter_map(|claim| match claim.origin() {
                ClaimOrigin::NonLive { index, kind } => Some((index, claim.range(), kind)),
                _ => None,
            })
            .collect::<Vec<_>>(),
        expected_capacity
    );
    drop(phase);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let native = file::libhdf5_open(&path);
    if paged {
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
    } else {
        assert_eq!(native.member_names().unwrap(), ["data"]);
        assert_eq!(
            native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
            [7, 11, 13]
        );
    }
    native.close().unwrap();
}
